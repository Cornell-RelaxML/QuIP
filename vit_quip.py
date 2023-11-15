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
# sys.path.append('../ViT-pytorch')
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from utils.train_utils import AverageMeter, count_parameters, simple_accuracy
import timm
import imageNet_utils as datasets

from models.modeling import VisionTransformer, CONFIGS

from utils.construct_tff import construct_real_tff
import pickle as pkl

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

def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logging.info("***** Running Validation *****")
    logging.info("  Num steps = %d", len(test_loader))
    logging.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logging.info("\n")
    logging.info("Validation Results")
    logging.info("Global Steps: %d" % global_step)
    logging.info("Valid Loss: %2.5f" % eval_losses.avg)
    logging.info("Valid Accuracy: %2.5f" % accuracy)

    if writer is not None:
        print('writing')
        writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy

@torch.no_grad()
def quantize_vit(model, train_batch, dev, args):
    print('Starting ...')
    # layers = model.transformer.encoder.layer
    # for batch in dataloader:
    #     inps = model.transformer.embeddings(batch[0].to(device))
    #     break

    layers = model.blocks
    inps = []
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            raise ValueError

    layers[0] = Catcher(layers[0])
    try:
        model(train_batch[0].to(dev))
    except ValueError:
        pass
    inps = inps[0]
    layers[0] = layers[0].module

    tffs = {}
    l_tff = 8

    outs = torch.zeros_like(inps)
    print('Ready.')

    quantizers = {}
    tff_rand_seeds = {}
    quantized_weights = {}
    Wiener_params = {}
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
                                               mse=False, 
                                               x_sigma= args.x_sigma)
            elif args.quant == 'nearest':
                quant_method[name] = Nearest(subset[name])
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False,
                                               x_sigma = args.x_sigma)
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
                                               mse=False, 
                                               x_sigma = args.x_sigma)

            if args.pre_tff:
                u_n = subset[name].weight.shape[0]
                v_n = subset[name].weight.shape[1]
                if u_n not in tffs:
                    k_tff = int(u_n // l_tff * args.tff_redundancy)
                    tffs[u_n] = construct_real_tff(k_tff, l_tff // 2, u_n // 2).to(dev)
                if v_n not in tffs:
                    k_tff = int(v_n // l_tff * args.tff_redundancy)
                    tffs[v_n] = construct_real_tff(k_tff, l_tff // 2, v_n // 2).to(dev)
                g_u = torch.Generator() # use this to store the seed for later
                u_seed = g_u.seed()
                rand_mat_u = torch.randn((u_n, u_n), generator=g_u)
                Q_u, _ = torch.linalg.qr(rand_mat_u)
                g_v = torch.Generator() # use this to store the seed for later
                v_seed = g_v.seed()
                rand_mat_v = torch.randn((v_n, v_n), generator=g_v)
                Q_v, _ = torch.linalg.qr(rand_mat_v)
                tff_rand_seeds[f'quantized model.decoder.layers.{i}.{name}'] = {'u_seed': u_seed, 'v_seed':v_seed}
                quant_method[name].U = tffs[u_n].view(-1, u_n) @ Q_u.T.to(dev)
                quant_method[name].V = tffs[v_n].view(-1, v_n) @ Q_v.T.to(dev)

            quant_method[name].name = name

        def add_batch(name):

            def tmp(_, inp, out):
                quant_method[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in subset:
            print(name, 'atttaching hook')
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.train_batch_size):
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
            # if args.pre_tff:
            #     clean_W = quant_method[name].layer.weight.data.clone()
            #     quant_method[name].layer.weight.data = quant_method[name].tff @ clean_W
            if quant_method[name].nsamples == 0:
                print(name)
                breakpoint()
                continue
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

            quantizers['model.decoder.layers.%d.%s' %
                        (i, name)] = quant_method[name].quantizer
            quantized_weights['model.decoder.layers.%d.%s' %
                        (i, name)] = quant_method[name].layer.weight.data.clone().cpu()

            # log the layer name
            logging.info(f'quantized model.decoder.layers.{i}.{name}')
            # apply the Weiner filter
            wiener_params_i = None
            # if args.pre_tff:
            #     if args.wiener_filt_en:
            #         wiener_params_i = quant_method[name].apply_weiner_filter(clean_W, args.Weiner_m_diag_rank)
            #     elif args.clamp_noise_filt_en:
            #         # implement the Wiener filter based on the clamped noise
            #         clamped_projs = quant_method[name].clamped_proj
            #         x = clean_W
            #         z = clamped_projs
            #         a = quant_method[name].tff @ x 
            #         n = z - a
            #         var_x = x.var()
            #         var_n = n.var()
            #         Wiener_F = (var_x * quant_method[name].tff.T) @ torch.linalg.pinv(var_x * quant_method[name].tff @ quant_method[name].tff.T + var_n)
            #         k=args.Weiner_m_diag_rank
            #         if k != 0:
            #             num_samples = clean_W.shape[1]
            #             Rxz = clean_W @ clamped_projs.T / num_samples
            #             Rzz = clamped_projs @ clamped_projs.T / num_samples
            #             full_Wiener_F = Rxz @ torch.linalg.pinv(Rzz)
            #             Wiener_residue = full_Wiener_F - Wiener_F
            #             U, S, Vt = torch.linalg.svd(Wiener_residue)
            #             Wiener_res_approx = torch.matmul(U[:, :k], torch.matmul(torch.diag(S[:k]), Vt[:k, :]))
            #         else:
            #             Wiener_res_approx = 0
            #         Wiener_F = Wiener_F + Wiener_res_approx
            #         # Wiener_F = Wiener_res_approx

            #         quant_method[name].layer.weight.data = Wiener_F @ quant_method[name].layer.weight.data
            #     else:
            #         quant_method[name].layer.weight.data = quant_method[name].tff.T @ quant_method[name].layer.weight.data



            Wiener_params['model.decoder.layers.%d.%s' %
                        (i, name)] = wiener_params_i

                # # run the linear quadratic program
                # import cvxpy as cp

                # xs = []
                # qw = quant_method[name].layer.weight.data / quant_method[name].quantizer.scale
                # solve_failed_count = 0
                # solve_failed_indices = []
                # for i in tqdm(range(clean_W.shape[1])):
                #     x = cp.Variable((clean_W.shape[0], 1))
                #     cost = cp.sum_squares(torch.zeros((clean_W.shape[0], 1)))
                #     constraints = [quant_method[name].tff.cpu() @ x <= (qw[:,i][...,None] + 1).cpu()]
                #     constraints += [quant_method[name].tff.cpu() @ x >= (qw[:,i][...,None] ).cpu()]
                #     prob = cp.Problem(cp.Minimize(cost), constraints)
                #     prob.solve(solver=cp.SCS, max_iters=2)

                #     print(x.value)
                #     if x.value is None:
                #         solve_failed_count += 1
                #         solve_failed_indices.append(i)

                #     # xs.append(torch.from_numpy(x.value))

                #     del x
                #     del cost
                #     del constraints
                #     del prob

                # final_x = torch.cat(xs, dim=1) * quant_method[name].quantizer.scale
                # breakpoint()
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
    return quantizers, errors, quantized_weights, Wiener_params, tff_rand_seeds

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

    return val_acc


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
    parser.add_argument('--tff_redundancy', type=float, default=1,
                        help="Redundancy in tffs")
    parser.add_argument('--wiener_filt_en', action='store_true',
                        help="enable the Wiener filter after TFF based quantization")
    parser.add_argument('--clamp_noise_filt_en', action='store_true',
                        help="Wiener filter based clamping noise filter")
    parser.add_argument('--num_vals', type=int, default=1,
                        help="num vals")
    parser.add_argument('--train_batch_path', type=str, default='./data/train_batch.pt',
                        help="path to the train batch")
    parser.add_argument('--dataset_path', type=str, default='/data/harsha/quantization/imagenet2012/',
                        help="path to the dataset; Dataset must contain val folder for testing")
    parser.add_argument("--timm_model_name", type=str, default='vit_base_patch16_224',
                        help="name of the model to be loaded from timm")
    parser.add_argument('--x_sigma', type=float, default=2,
                        help="x times sigma for symm scale")

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
                        help='qfn: a is default, b is sym incoherent based, s is tff symm scaling based')
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
            directory_path = os.path.join("output_new", f'{args.parent_dir}', f'wb{args.wbits}', f'{exp_name}_{current_datetime}')
            args.save_path = directory_path
        else:
            directory_path = args.save_path

        os.makedirs(directory_path, exist_ok=True)

        logging.basicConfig(filename= os.path.join(directory_path, 'log.log'), level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_values = {}

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

    if exp_name != 'debug_thread':
        with open(os.path.join(directory_path, 'args.pkl'), 'wb') as handle:
            pkl.dump(vars(args), handle)

    # args, model = setup(args)
    # model.eval()

    name = args.timm_model_name
    model = timm.create_model(name, pretrained=True).to(device)
    model.eval()
    num_params = sum([p.numel() for p in model.parameters()])
    print(f'num_params = {num_params/ 1e6}')
    logging.info(f'num_params = {num_params/ 1e6}')

    g=datasets.ViTImageNetLoaderGenerator(args.dataset_path,'imagenet',args.train_batch_size,args.eval_batch_size,16, kwargs={"model":model, "img_size":args.img_size})
    train_loader  = g.train_loader()
    train_batch = next(iter(train_loader))
    train_labels_used = train_batch[1]
    # train_batch = torch.load(args.train_batch_path)

    log_values['model_name'] = name
    log_values['num_params'] = num_params

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
        logging.info(f"Preprocessing flags: gptqH:{args.pre_gptqH}, rescale:{args.pre_rescale}, proj:{args.pre_proj}, proj_extra:{args.pre_proj_extra}, qfn:{args.qfn}")
        logging.info(f"using lazy_batch updates: {args.lazy_batch}")
        # LDL checks
        if ('ldl' in args.quant) and args.unbiased and (args.npasses > 0):
            print(f"LDL NOTE: unbiased + {args.npasses} npasses. NOT TRULY UNBIASED.")
            logging.info(f"LDL NOTE: unbiased + {args.npasses} npasses. NOT TRULY UNBIASED.")

        tick = time.time()
        quantizers, errors, quantized_weights, Wiener_params, tff_rand_seeds = quantize_vit(model, train_batch, args.device, args)
        # quantizers, errors = quantize_vit(model, test_loader, args.device, args)
        print(f'Total quant + H time elapsed: {time.time() - tick:.2f}s')
        print("")
        print(f'Proxy Summary: Qmethod:{args.quant}, Unbiased: {args.unbiased}, W:{args.wbits}, NPass:{args.npasses}')
        print('Quantization done.')
        print("")
        total_quant_time = time.time() - tick
        log_values['total_quant_time'] = total_quant_time
        logging.info(f'Total quant + H time elapsed: {total_quant_time:.2f}s')
        logging.info("")
        logging.info(f'Proxy Summary: Qmethod:{args.quant}, Unbiased: {args.unbiased}, W:{args.wbits}, NPass:{args.npasses}')
        logging.info('Quantization done.')
        logging.info("")

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
        # save the model
        torch.save(model.state_dict(), os.path.join(directory_path, 'Qmodel.pth'))
        mdict = {   'quantizers': quantizers,
                    'errors':errors,
                    'quantized_weights':quantized_weights,
                    'Wiener_params':Wiener_params, 
                    'tff_rand_seeds': tff_rand_seeds
                    }
        torch.save(mdict, os.path.join(directory_path, 'Qparams.pth'))

    if not args.proxy_only:
        # val_acc = valid(args, model, writer=None, test_loader=test_loader, global_step=0)
        # g=datasets.ViTImageNetLoaderGenerator('/data/harsha/quantization/imagenet2012','imagenet',args.train_batch_size,args.eval_batch_size,16, kwargs={"model":model})
        # test_loader = g.test_loader()
        test_loader = g.test_loader()
        for i in range(args.num_vals):
            val_acc = custom_val(model, test_loader, device)
            print(f'intermediate val acc = {val_acc}')
            logging.info(f'intermediate val acc = {val_acc}')


    # save the results
    if exp_name != 'debug_thread':
        log_values['val_acc'] = val_acc
        log_values['args'] = args
        with open(os.path.join(directory_path, 'log_values.pkl'), 'wb') as handle:
            pkl.dump(log_values, handle)

        import csv
        results  = [name, num_params/1e6, args.wbits, val_acc]
        csv_file_path = os.path.join("output_new", f'{args.parent_dir}', f'wb{args.wbits}','results.csv')
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(results)


# # get the different tff_n values for different models
# tff_ns_all = {  'vit_tiny_patch16_224': [576, 192, 768],
#             'vit_small_patch16_224': [1152, 384, 1536],
#             'vit_small_patch32_224': [1152, 384, 1536],
#             'vit_base_patch16_224': [2304, 768, 3072],
#             'deit_tiny_patch16_224': [576, 192, 768],
#             'deit_small_patch16_224': [1152, 384, 1536],
#             'deit_base_patch16_224': [2304, 768, 3072], 
#             'vit_huge_patch14_clip_224.laion2b_ft_in1k': [3840, 1280, 5120], 
#             'vit_large_patch16_224.augreg_in21k_ft_in1k': [3072, 1024, 4096],
#             'beit_base_patch16_384.in22k_ft_in22k_in1k': [2304, 768, 3072],
#             'beit_large_patch16_512.in22k_ft_in22k_in1k': [3072, 1024, 4096],
#             'beit_base_patch16_224.in22k_ft_in22k_in1k': [2304, 768, 3072],
#             'beit_large_patch16_224.in22k_ft_in22k_in1k': [3072, 1024, 4096],
#             'beitv2_base_patch16_224.in1k_ft_in1k': [2304, 768, 3072],
#             }
