import time

import torch
import torch.nn as nn

from gptq import *
from bal import Balance
from near import Nearest
from modelutils import *
from quant import *

from tqdm import tqdm

from opt import get_opt


@torch.no_grad()
def opt_sequential_saveH(model, dataloader, dev, args):
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
            elif args.quant == 'gptq_updown':
                quant_method[name] = GPTQ_UD(subset[name])
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(args.wbits,
                                               perchannel=True,
                                               sym=False,
                                               qfn=args.qfn,
                                               mse=False)
            elif args.quant in ['bitbal','parbal','allbal']:
                quant_method[name] = Balance(subset[name])
                quant_method[name].configure(
                                    args.quant,
                                    args.wbits, 
                                    args.npasses,
                                    unbiased=False)
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

        # Quantize Weights and Save Hessian
        for name in subset:
            # print(i, name)
            # print('Quantizing ...')
            quant_method[name].preproc(
                                preproc_gptqH=True, percdamp=args.percdamp,
                                preproc_rescale=False, 
                                preproc_proj=False, preproc_proj_extra=0)
            if args.quant == 'gptq':
                quant_method[name].fasterquant(groupsize=args.groupsize, copy_H=True)
                quantizers['model.decoder.layers.%d.%s' %
                           (i, name)] = quant_method[name].quantizer
            if args.quant == 'gptq_updown':
                quant_method[name].fasterquant_updown(groupsize=args.groupsize)
            elif args.quant in ['bitbal','parbal','allbal']:
                quant_method[name].fasterquant()
            elif args.quant == 'nearest':
                quant_method[name].fasterquant()

            fname = f'{args.save}/H_model.decoder.layers.{i}.{name}.pt'
            torch.save(quant_method[name].H.cpu(), fname)

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
    print(f'Total quant time: {sum(times):.2f}s')

    return 


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
                        choices=['bitbal', 'parbal', 'allbal', 'nearest', 'gptq', 'gptq_updown'],
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
        default=1,
        help='number passes to repeat balance loop over 1-d.')
    parser.add_argument(
        '--groupsize',
        type=int,
        default=-1,
        help='Groupsize to use for quantization; default uses full row.')
    # parser.add_argument(
    #     '--pre_gptqH',
    #     action='store_true',
    #     help='preprocessing')
    # parser.add_argument(
    #     '--pre_rescale',
    #     action='store_true',
    #     help='preprocessing')
    # parser.add_argument(
    #     '--pre_proj',
    #     action='store_true',
    #     help='preprocessing')
    parser.add_argument('--qfn',
                        type=str,
                        default='a',
                        help='qfn: a is default, b is sym incoherent based')
    parser.add_argument('--save',
                        type=str,
                        default='',
                        help='Save quantized checkpoint under this name.')
    # parser.add_argument('--load',
    #                     type=str,
    #                     default='',
    #                     help='Load quantized model.')
    # parser.add_argument('--benchmark',
    #                     type=int,
    #                     default=0,
    #                     help='Number of tokens to use for benchmarking.')
    # parser.add_argument(
    #     '--check',
    #     action='store_true',
    #     help=
    #     'Whether to compute perplexity during benchmarking for verification.')
    # parser.add_argument(
    #     '--saveH',
    #     action='store_true',
    #     help=
    #     'Saves Hessians')

    args = parser.parse_args()
    assert args.save

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
        print(f"preprocessing flags: gptqH:True, rescale:False, proj:False, qfn:{args.qfn}")

        tick = time.time()
        opt_sequential_saveH(model, dataloader, DEV, args)
        print("Done save H")
        print("")

