import time

import torch
import torch.nn as nn

from gptq import *
from bal import Balance
from near import Nearest
from modelutils import *
from quant import *

from tqdm import tqdm

import logging
import datetime
from datetime import timedelta
import os
import random
import sys
sys.path.append('../ViT-pytorch')
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from utils.train_utils import AverageMeter, count_parameters, simple_accuracy

from models.modeling import VisionTransformer, CONFIGS

from utils.construct_tff import construct_real_tff
from vit import valid

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100
    elif 'dogs' in args.dataset:
        num_classes = 10
    elif 'trucks' in args.dataset:
        num_classes = 6
    elif 'cats' in args.dataset:
        num_classes = 6
    elif 'birds' in args.dataset:
        num_classes = 10
    elif 'snakes' in args.dataset:
        num_classes = 10

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.load_state_dict(torch.load(args.ckpt_path))    
    model.to(args.device)
    num_params = count_parameters(model)

    logging.info("{}".format(config))
    logging.info("Training parameters %s", args)
    logging.info("Total Parameter: \t%2.1fM" % num_params)
    print(f'{num_params = }M')
    return args, model

@torch.no_grad()
def quantize_vit(model, dataloader, dev, args):
    print('Starting ...')
    # layers = model.transformer.encoder.layer
    # for batch in dataloader:
    #     inps = model.transformer.embeddings(batch[0].to(device))
    #     break

    layers = model.blocks
    dtype = next(iter(model.parameters())).dtype
    inps = []
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            break
    inps = inps[0]
    layers[0] = layers[0].module

    tffs = {}
    k_attn = int(96 * args.tff_redundancy)
    l_attn = 8
    n_attn = 768
    tffs[n_attn] = construct_real_tff(k_attn, l_attn // 2, n_attn // 2).to(dev)

    k_mlp = int(384 * args.tff_redundancy)
    l_mlp = 8
    n_mlp = 3072
    tffs[n_mlp] = construct_real_tff(k_mlp, l_mlp // 2, n_mlp // 2).to(dev)

    k_mlp = int(288 * args.tff_redundancy)
    l_mlp = 8
    n_mlp = 2304
    tffs[n_mlp] = construct_real_tff(k_mlp, l_mlp // 2, n_mlp // 2).to(dev)

    outs = torch.zeros_like(inps)
    print('Ready.')

    WFs = {}
    WFs[n_attn] = {'Rxz':torch.zeros(n_attn, n_attn), 'Rzz':torch.zeros(n_attn, n_attn), 'first': True, 'num_samples': 0}
    WFs[n_mlp] = {'Rxz':torch.zeros(n_mlp, n_mlp), 'Rzz':torch.zeros(n_mlp, n_mlp), 'first': True, 'num_samples': 0}

    quantizers = {}
    errors, Hmags, times = [], [], []
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        quant_method = {}
        # Initialize Quant Method and Compute H
        for name in subset:
            if args.quant == 'gptq':
                quant_method[name] = GPTQ(subset[name])
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False)
            elif args.quant == 'nearest':
                quant_method[name] = Nearest(subset[name])
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False)
            elif args.quant in ['allbal','ldlq','ldlqRG','ldlbal_admm']:
                quant_method[name] = Balance(subset[name])
                quant_method[name].configure(
                                    args.quant,
                                    args.wbits, 
                                    args.npasses,
                                    unbiased=args.unbiased)
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False)
                tff_n = subset[name].weight.shape[0]
                quant_method[name].tff = tffs[tff_n].view(-1, tff_n)

        def add_batch(name):

            def tmp(_, inp, out):
                quant_method[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0))[0]
        for h in handles:
            h.remove()
        # (H / nsamples).to(torch.float32)
        for name in subset:
            quant_method[name].post_batch()

        # Quantize Weights
        for name in subset:
            # print(i, name)
            # print('Quantizing ...')

            # project onto the frames
            if args.pre_tff:
                clean_W = quant_method[name].layer.weight.data.clone()
                quant_method[name].layer.weight.data = quant_method[name].tff @ clean_W
            quant_method[name].preproc(
                                preproc_gptqH=args.pre_gptqH, percdamp=args.percdamp,
                                preproc_rescale=args.pre_rescale, 
                                preproc_proj=args.pre_proj, preproc_proj_extra=args.pre_proj_extra)
            if args.quant == 'gptq':
                quant_method[name].fasterquant(groupsize=args.groupsize)
            elif args.quant in ['allbal','ldlq','ldlqRG','ldlbal_admm']:
                quant_method[name].fasterquant(lazy_batch=args.lazy_batch)
            elif args.quant == 'nearest':
                quant_method[name].fasterquant()

            # apply the Weiner filter
            if args.pre_tff:
                if args.wiener_filt_en:
                    quant_method[name].apply_weiner_filter(clean_W, args.Weiner_m_diag_rank)
                else:
                    quant_method[name].layer.weight.data = quant_method[name].tff.T @ quant_method[name].layer.weight.data

            quantizers['model.decoder.layers.%d.%s' %
                        (i, name)] = quant_method[name].quantizer
            # 

            errors.append(quant_method[name].error)
            times.append(quant_method[name].time)
            Hmags.append(quant_method[name].Hmag)
            quant_method[name].free()

        
        # outs = layer(inps)[0]
        outs = layer(inps)

        del layer
        del quant_method
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    print(f'Total quant time: {sum(times):.2f}s')
    return quantizers, errors


# TODO: perform packing on GPU
def opt_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale,
                           quantizers[name].zero)
    print('Done.')
    return model


def load_quant3(model, checkpoint):
    from transformers import OPTConfig, OPTForCausalLM
    config = OPTConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = OPTForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in [
            'model.decoder.project_out', 'model.decoder.project_in', 'lm_head'
    ]:
        if name in layers:
            del layers[name]
    make_quant3(model, layers)

    print('Loading model ...')
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = model.config.max_position_embeddings
    print('Done.')

    return model

def load_quant(model, checkpoint):
    from transformers import OPTConfig, OPTForCausalLM
    config = OPTConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = OPTForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in [
            'model.decoder.project_out', 'model.decoder.project_in', 'lm_head'
    ]:
        if name in layers:
            del layers[name]
    # make_quant3(model, layers)


    print('Loading model ...')
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = model.config.max_position_embeddings
    print('Done.')

    return model


def opt_multigpu(model, gpus):
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(
        gpus[0])
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
        gpus[0])
    if hasattr(model.model.decoder,
               'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(
            gpus[0])
    if hasattr(model.model.decoder,
               'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(
            gpus[-1])
    if hasattr(model.model.decoder,
               'final_layer_norm') and model.model.decoder.final_layer_norm:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(
            gpus[-1])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])

    cache = {'mask': None}

    class MoveModule(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache['mask'] is None or cache['mask'].device != self.dev:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.decoder.layers
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))

    model.gpus = gpus


def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    cache = {'past': None}

    def clear_past(i):

        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None

        return tmp

    for i, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(input_ids[:, i].reshape(-1),
                        past_key_values=cache['past'],
                        attention_mask=attention_mask[:, :(i + 1)].reshape(
                            (1, -1)))
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV),
                            input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        import numpy as np
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())

def custom_val(net, test_set, device):
    all_labels = []
    all_preds = []
    num_examples = 0
    correct_counts = 0
    for batch_idx, (images, labels) in tqdm(enumerate(test_set)):
        images, labels = images.to(device), labels.to(device)
        logits = net(images)
        preds = torch.argmax(logits, dim=-1)

        all_labels.append(labels.cpu().unbind())
        all_preds.append(preds.cpu().unbind())
        # all_preds.append(preds.cpu().item())
        num_examples += len(labels)
        correct_counts += sum(preds == labels)

    val_acc = correct_counts  / num_examples
    print(f'{name = }, {val_acc = }')
    logging.info(f'{name = }, {val_acc = }')


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--exp_name", type=str, default='debug_thread',
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--model_type", type=str, default='ViT-B_16',
                        help="Type of the ViT you are using")
    parser.add_argument("--img_size", type=int, default=224,
                        help="resolution of the square images")
    parser.add_argument("--pretrained_dir", type=str, default='checkpoint/ViT-B_16-224.npz',
                        help="resolution of the square images")
    parser.add_argument("--device", type=str, default=None,
                        help="device that you want to run on; currently this is just a placeholder")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="relevant when you have multiple devices")
    parser.add_argument("--train_batch_size", type=int, default=128,
                        help="training batch size; Here it serves as nsamples as well")
    parser.add_argument("--nsamples", type=int, default=128,
                        help="nsamples to be used for quantization")
    parser.add_argument("--eval_batch_size", type=int, default=128,
                        help="eval batch size")
    parser.add_argument("--dataset", type=str, default='inet1k_birds',
                        help="name of the dataset")
    parser.add_argument("--dataset_dir", type=str, default='data/inet1k_classes/birds',
                        help="path to the dataset")
    parser.add_argument("--ckpt_path", type=str, default='output/inet1k_birds-2023-10-17-03-04-30/inet1k_birds_final_ckpt.bin',
                        help="path to the saved checkpoint of the model")
    parser.add_argument("--coef_est_type", type=str, default='weiner', choices = ['weiner', 'naive'],
                        help="how to estimate the weights from the quantized versions")
    parser.add_argument("--save_path", type=str, default=None, 
                        help="provide the savepath; otherwise a cat of exp_name and current time will be used")
    parser.add_argument("--parent_dir", type=str, default=None, 
                        help="parent dir for storing the results")
    parser.add_argument("--Weiner_m_diag_rank", type=int, default=3,
                        help="set the rank for the LowRank approximation of the residue after (Weiner - diag)")
    parser.add_argument('--tff_redundancy', type=int, default=1,
                        help="Redundancy in tffs")
    parser.add_argument('--wiener_filt_en', action='store_true',
                        help="enable the Wiener filter after TFF based quantization")

    parser.add_argument(
        '--percdamp',
        type=float,
        default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--quant',
                        choices=['allbal', 
                        'ldlq', 'ldlqRG', 'ldlbal_admm', 
                        'nearest', 'gptq'],
                        default='nearest',
                        help='Which quantization method to use.')
    parser.add_argument(
        '--wbits',
        type=int,
        default=16,
        choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument(
        '--npasses',
        type=int,
        default=0,
        help='number passes to repeat balance loop over 1-d.')
    parser.add_argument(
        '--groupsize',
        type=int,
        default=-1,
        help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument(
        '--pre_tff',
        action='store_true',
        help='preprocessing')
    parser.add_argument(
        '--pre_gptqH',
        action='store_true',
        help='preprocessing')
    parser.add_argument(
        '--pre_rescale',
        action='store_true',
        help='preprocessing')
    parser.add_argument(
        '--pre_proj',
        action='store_true',
        help='preprocessing')
    parser.add_argument(
        '--pre_proj_extra',
        type=int,
        default=0,
        choices=[0, 1, 2],
        help='Extra options to control pre_proj step.')
    parser.add_argument('--qfn',
                        type=str,
                        default='a',
                        help='qfn: a is default, b is sym incoherent based')
    parser.add_argument('--save',
                        type=str,
                        default='',
                        help='Save quantized checkpoint under this name.')
    parser.add_argument('--load',
                        type=str,
                        default='',
                        help='Load quantized model.')
    # parser.add_argument('--benchmark',
    #                     type=int,
    #                     default=0,
    #                     help='Number of tokens to use for benchmarking.')
    parser.add_argument(
        '--check',
        action='store_true',
        help=
        'Whether to compute perplexity during benchmarking for verification.')
    parser.add_argument(
        '--proxy_only',
        action='store_true',
        help=
        'Only compute proxy objective (w^T H w)')
    parser.add_argument(
        '--unbiased',
        action='store_true',
        help='unbiased')
    parser.add_argument(
        '--incoh_processing',
        action='store_true',
        help='incoherence processing')
    parser.add_argument(
        '--lazy_batch',
        action='store_true',
        help='lazy batch updates in blocks as used in OPTQ')

    args = parser.parse_args()
    # defaults to incoherence processing
    if args.incoh_processing:
        args.pre_gptqH   = True
        args.pre_rescale = True
        args.pre_proj    = True
        args.proj_extra  = 1
        args.qfn         = 'b'

    # logging 
    exp_name = args.exp_name # 'mlp_attn_quant_weiner_full'
    if exp_name != 'debug_thread':
        if args.save_path is None:
            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            directory_path = os.path.join("output", f'{args.parent_dir}', f'{exp_name}_{current_datetime}')
            args.save_path = directory_path
        else:
            directory_path = args.save_path

        os.makedirs(directory_path, exist_ok=True)

        logging.basicConfig(filename= os.path.join(directory_path, 'log.log'), level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        print(f'{args.n_gpu = }')
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Set seed
    set_seed(args)

    print('args = ')
    logging.info('args = ')
    for k,v in vars(args).items():
        print(f'{k}: {v}')
        logging.info(f'{k}: {v}')

    if args.load:
        # need to fix this
        model = load_quant(args.model, args.load)
        model.eval()
    else:
        # args, model = setup(args)
        # model.eval()
        import timm
        import imageNet_utils as datasets

        name = 'vit_base_patch16_224'
        model = timm.create_model(name, pretrained=True).to(device)
        model.eval()
        num_params = sum([p.numel() for p in model.parameters()])
        print(f'num_params = {num_params/ 1e6}')
        logging.info(f'num_params = {num_params/ 1e6}')
        g=datasets.ViTImageNetLoaderGenerator('/data/harsha/quantization/imagenet2012','imagenet',args.train_batch_size,args.train_batch_size,16, kwargs={"model":model})
        test_loader = g.test_loader()

        # Prepare dataset
        # train_loader, test_loader = get_loader(args)
        # val_acc = valid(args, model, writer=None, test_loader=test_loader, global_step=0)
        # print(f'initial val acc = {val_acc}')
        # logging.info(f'initial val acc = {val_acc}')

        if args.wbits < 16:
            # Preprocessing flags
            # if args.qfn=='b': assert args.pre_proj is True
            print(f"Preprocessing flags: gptqH:{args.pre_gptqH}, rescale:{args.pre_rescale}, proj:{args.pre_proj}, proj_extra:{args.pre_proj_extra}, qfn:{args.qfn}")
            print(f"using lazy_batch updates: {args.lazy_batch}")
            # LDL checks
            if ('ldl' in args.quant) and args.unbiased and (args.npasses > 0):
                print(f"LDL NOTE: unbiased + {args.npasses} npasses. NOT TRULY UNBIASED.")

            tick = time.time()
            # quantizers, errors = quantize_vit(model, train_loader, args.device, args)
            quantizers, errors = quantize_vit(model, test_loader, args.device, args)
            print(f'Total quant + H time elapsed: {time.time() - tick:.2f}s')
            print("")
            print(f'Proxy Summary: Qmethod:{args.quant}, Unbiased: {args.unbiased}, W:{args.wbits}, NPass:{args.npasses}')
            print('Quantization done.')
            print("")

    # if args.benchmark:
    #     gpus = [
    #         torch.device('cuda:%d' % i)
    #         for i in range(torch.cuda.device_count())
    #     ]
    #     if len(gpus) > 1:
    #         opt_multigpu(model, gpus)
    #     else:
    #         model = model.to(DEV)
    #     if args.benchmark:
    #         input_ids = next(iter(dataloader))[0][:, :args.benchmark]
    #         benchmark(model, input_ids, check=args.check)
    # if args.load:
    #     exit()

    if exp_name != 'debug_thread':
    #     opt_pack3(model, quantizers)
        torch.save(model.state_dict(), os.path.join(directory_path, 'Qmodel.pth'))

    if not args.proxy_only:
        # val_acc = valid(args, model, writer=None, test_loader=test_loader, global_step=0)
        val_acc = custom_val(model, test_loader, device)
        print(f'intermediate val acc = {val_acc}')
        logging.info(f'intermediate val acc = {val_acc}')

