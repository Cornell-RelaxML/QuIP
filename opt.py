import time

import torch
import torch.nn as nn

from gptq import *
from bal import Balance
from near import Nearest
from modelutils import *
from quant import *

from tqdm import tqdm

def get_opt(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model


@torch.no_grad()
def opt_sequential(model, dataloader, dev, args):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
        dev)
    if hasattr(model.model.decoder,
               'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(
            dev)
    if hasattr(model.model.decoder,
               'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size),
                       dtype=dtype,
                       device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu(
    )
    if hasattr(model.model.decoder,
               'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder,
               'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

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

        def add_batch(name):

            def tmp(_, inp, out):
                quant_method[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0),
                            attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()
        # (H / nsamples).to(torch.float32)
        for name in subset:
            quant_method[name].post_batch()

        # Quantize Weights
        for name in subset:
            # print(i, name)
            # print('Quantizing ...')
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

            errors.append(quant_method[name].error)
            times.append(quant_method[name].time)
            Hmags.append(quant_method[name].Hmag)
            quant_method[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0),
                            attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del quant_method
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    # print("errors")
    # print(errors)
    # print("Hmags")
    # print(Hmags)
    print(f'Total quant time: {sum(times):.2f}s')

    return quantizers, errors


@torch.no_grad()
def opt_eval(model, testenc, dev):
    # print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
        dev)
    if hasattr(model.model.decoder,
               'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(
            dev)
    if hasattr(model.model.decoder,
               'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size),
                       dtype=dtype,
                       device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu(
    )
    if hasattr(model.model.decoder,
               'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder,
               'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in tqdm(range(len(layers))):
        # print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0),
                            attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(
            dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(
            dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:,
                               (i * model.seqlen):((i + 1) * model.seqlen)][:,
                                                                            1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


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


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument('model',
                        type=str,
                        help='OPT model to load; pass `facebook/opt-X`.')
    parser.add_argument('dataset',
                        type=str,
                        choices=['wikitext2', 'ptb', 'c4'],
                        help='Where to extract calibration data from.')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples',
                        type=int,
                        default=128,
                        help='Number of calibration data samples.')
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

    if args.load:
        model = load_quant(args.model, args.load)
        model.eval()
    else:
        model = get_opt(args.model)
        model.eval()

        dataloader, _ = get_loaders(args.dataset,
                                            nsamples=args.nsamples,
                                            seed=args.seed,
                                            model=args.model,
                                            seqlen=model.seqlen)

        if args.wbits < 16:
            # Preprocessing flags
            if args.qfn=='b': assert args.pre_proj is True
            print(f"Preprocessing flags: gptqH:{args.pre_gptqH}, rescale:{args.pre_rescale}, proj:{args.pre_proj}, proj_extra:{args.pre_proj_extra}, qfn:{args.qfn}")
            print(f"using lazy_batch updates: {args.lazy_batch}")
            # LDL checks
            if ('ldl' in args.quant) and args.unbiased and (args.npasses > 0):
                print(f"LDL NOTE: unbiased + {args.npasses} npasses. NOT TRULY UNBIASED.")

            tick = time.time()
            quantizers, errors = opt_sequential(model, dataloader, DEV, args)
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

    if args.save:
    #     opt_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save)

    if not args.proxy_only:
        # for dataset in ['wikitext2', 'ptb', 'c4']:
        for dataset in ['wikitext2', 'ptb-new', 'c4-new']:
            dataloader, testloader = get_loaders(dataset,
                                                seed=args.seed,
                                                model=args.model,
                                                seqlen=model.seqlen)
            print(dataset)
            opt_eval(model, testloader, DEV)

