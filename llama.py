import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import quant
import logging
from datetime import timedelta

# from gptq import GPTQ, Observer
from gptq import *
from bal import Balance
from near import Nearest
from modelutils import *
from quant import *
from method import rand_ortho_butterfly

from utils.llama_utils import get_wikitext2, get_ptb, get_c4, get_ptb_new, get_c4_new, get_loaders, gen_conditions
from utils.llama_export import export_quant_table
from datautils import set_seed
from texttable import Texttable
from tqdm import tqdm

from utils.construct_tff import construct_real_tff
import pickle as pkl
from typing import List, Optional, Tuple, Union
import warnings

import subprocess

@torch.no_grad()
def llama_emb_quant(subset, dev, args):
    ######################################################
    # quantize embedding
    ######################################################
    name = 'emb'
    quant_method = {}
    if args.emb_quant == 'gptq':
        quant_method[name] = GPTQ(subset[name])
        quant_method[name].quantizer = Quantizer()
        quant_method[name].quantizer.configure(args.emb_wbits,
                                       perchannel=True,
                                       sym=False,
                                       qfn=args.emb_qfn,
                                       mse=False, 
                                       x_sigma= args.x_sigma)
    elif args.emb_quant == 'nearest':
        quant_method[name] = Nearest(subset[name])
        quant_method[name].quantizer = Quantizer()
        quant_method[name].quantizer.configure(args.emb_wbits,
                                       perchannel=True,
                                       sym=False,
                                       qfn=args.emb_qfn,
                                       mse=False, 
                                       x_sigma= args.x_sigma)
    elif args.emb_quant in ['allbal','ldlq','ldlqRG','ldlbal_admm']:
        quant_method[name] = Balance(subset[name])
        quant_method[name].configure(
                            args.emb_quant,
                            args.emb_wbits, 
                            args.npasses,
                            unbiased=args.unbiased)
        quant_method[name].quantizer = Quantizer()
        quant_method[name].quantizer.configure(args.emb_wbits,
                                       perchannel=True,
                                       sym=False,
                                       qfn=args.emb_qfn,
                                       mse=False, 
                                       x_sigma= args.x_sigma)

        if args.pre_tff:
            u_n = subset[name].weight.shape[0]
            v_n = subset[name].weight.shape[1]

            l_tff = u_n // 16
            k_tff = round(u_n // l_tff * args.emb_redundancy)
            tffs_u = construct_real_tff(k_tff, l_tff // 2, u_n // 2)

            l_tff = v_n // 16
            k_tff = round(v_n // l_tff * args.emb_redundancy)
            tffs_v = construct_real_tff(k_tff, l_tff // 2, v_n // 2)

            g_u = torch.Generator() # use this to store the seed for later
            g_u.manual_seed(args.seed + 98776)
            u_seed = g_u.seed()
            rand_mat_u = torch.randn((u_n, u_n), generator=g_u)
            Q_u, _ = torch.linalg.qr(rand_mat_u)
            g_v = torch.Generator() # use this to store the seed for later
            g_u.manual_seed(args.seed + 98777)
            v_seed = g_v.seed()
            rand_mat_v = torch.randn((v_n, v_n), generator=g_v)
            Q_v, _ = torch.linalg.qr(rand_mat_v)
            quant_method[name].U = tffs_u.view(-1, u_n) @ Q_u.T
            quant_method[name].V = tffs_v.view(-1, v_n) @ Q_v.T

    quant_method[name].H = torch.eye(subset[name].weight.shape[1], dtype=torch.float32, device=dev)

    quant_method[name].name = name

    # quant_method[name].preproc(
    #                     preproc_gptqH=args.pre_gptqH, percdamp=args.percdamp,
    #                     preproc_rescale=args.pre_rescale, 
    #                     preproc_proj=args.pre_proj, preproc_proj_extra=args.pre_proj_extra)
    quant_method[name].preproc(
                        preproc_gptqH=args.emb_pre_gptqH, percdamp=args.percdamp,
                        preproc_rescale=args.emb_pre_rescale, 
                        preproc_proj=args.emb_pre_proj, preproc_proj_extra=args.pre_proj_extra)
    print(f'$$$$$$$$$$$$$$$$$ preproc done $$$$$$$$$$$$$$$$$$')
    if args.emb_quant == 'gptq':
        quant_method[name].fasterquant(groupsize=args.groupsize)
    elif args.emb_quant in ['allbal','ldlq','ldlqRG','ldlbal_admm']:
        quant_method[name].fasterquant(lazy_batch=args.lazy_batch)
    elif args.emb_quant == 'nearest':
        quant_method[name].fasterquant()
    # quantizers['model.decoder.layers.%d.%s' %
    #             (i, name)] = quant_method[name].quantizer.to('cpu')
    # errors.append(quant_method[name].error)
    # times.append(quant_method[name].time)
    # Hmags.append(quant_method[name].Hmag)

    quant_method[name].free()
    del quant_method


@torch.no_grad()
def llama_sequential(model, dataloader, dev, seed = 0):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    # Do the embed quantization
    if args.emb_quant_en:
        name = 'emb'
        subset = {name: model.model.embed_tokens}
        llama_emb_quant(subset, dev, args)
        print('embed quant done')

        if args.emb_quant_only:
            return {}


    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    tffs = {}
    print('Ready.')

    quantizers = {}
    tff_rand_seeds = {}
    errors, Hmags, times = [], [], []
    if args.observe:
        observer = Observer()
    else:
        observer = None
    lin_count = 0
    for i in range(len(layers)):

        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+------------------+--------------+------------+-----------+-------+')
        print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
        print('+==================+==============+============+===========+=======+')

        class dual_gpu_layer(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                # send attention to cuda 0, mlp to cuda 1
                self.module.self_attn = self.module.self_attn.to('cuda:0')
                self.module.input_layernorm = self.module.input_layernorm.to('cuda:0')

                self.module.mlp = self.module.mlp.to('cuda:1')
                self.module.post_attention_layernorm = self.module.post_attention_layernorm.to('cuda:1')

            def forward(
                self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: Optional[bool] = False,
                use_cache: Optional[bool] = False,
                **kwargs,
            ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
                """
                Args:
                    hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
                    attention_mask (`torch.FloatTensor`, *optional*):
                        attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                        query_sequence_length, key_sequence_length)` if default attention is used.
                    output_attentions (`bool`, *optional*):
                        Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                        returned tensors for more detail.
                    use_cache (`bool`, *optional*):
                        If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                        (see `past_key_values`).
                    past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
                """
                if "padding_mask" in kwargs:
                    warnings.warn(
                        "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
                    )

                # device 0
                hidden_states = hidden_states.to('cuda:0')
                residual = hidden_states

                hidden_states = self.module.input_layernorm(hidden_states)

                # Self Attention
                hidden_states, self_attn_weights, present_key_value = self.module.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs,
                )
                hidden_states = residual + hidden_states

                # Fully Connected
                # device 1
                hidden_states = hidden_states.to('cuda:1')
                residual = hidden_states
                hidden_states = self.module.post_attention_layernorm(hidden_states)
                hidden_states = self.module.mlp(hidden_states)
                hidden_states = residual + hidden_states

                outputs = (hidden_states,)

                if output_attentions:
                    outputs += (self_attn_weights,)

                if use_cache:
                    outputs += (present_key_value,)

                return outputs


        layer = dual_gpu_layer(layers[i])

        print(f' sent layer to dev')
        logging.info(f' sent layer to dev')

        full = find_layers(layer)
        if args.true_sequential:
            sequential = [['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'], ['self_attn.o_proj'], ['mlp.up_proj', 'mlp.gate_proj'], ['mlp.down_proj']]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            print(f'layer[{i}].{names}')
            logging.info(f'layer[{i}].{names}')
            lin_count += 1
            subset = {n: full[n] for n in names}
            quant_method = {}
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
                        l_tff = u_n // 16
                        k_tff = round(u_n // l_tff * args.tff_redundancy)
                        logging.info(f'layer {i}: {name}, red = {args.tff_redundancy}, {k_tff = }, {l_tff = }, {u_n = }')
                        print(f'layer {i}: {name}, red = {args.tff_redundancy}, {k_tff = }, {l_tff = }, {u_n = }')
                        tffs[u_n] = construct_real_tff(k_tff, l_tff // 2, u_n // 2)
                    if v_n not in tffs:
                        l_tff = v_n // 16
                        k_tff = round(v_n // l_tff * args.tff_redundancy)
                        logging.info(f'layer {i}: {name}, red = {args.tff_redundancy}, {k_tff = }, {l_tff = }, {v_n = }')
                        print(f'layer {i}: {name}, red = {args.tff_redundancy}, {k_tff = }, {l_tff = }, {v_n = }')
                        tffs[v_n] = construct_real_tff(k_tff, l_tff // 2, v_n // 2)
                    g_u = torch.Generator() # use this to store the seed for later
                    g_u.manual_seed(seed + lin_count)
                    u_seed = g_u.initial_seed()
                    rand_mat_u = torch.randn((u_n, u_n), generator=g_u)
                    Q_u, _ = torch.linalg.qr(rand_mat_u)
                    g_v = torch.Generator() # use this to store the seed for later
                    g_v.manual_seed(seed + lin_count)
                    v_seed = g_v.initial_seed()
                    rand_mat_v = torch.randn((v_n, v_n), generator=g_v)
                    Q_v, _ = torch.linalg.qr(rand_mat_v)
                    tff_rand_seeds[f'quantized model.decoder.layers.{i}.{name}'] = {'u_seed': u_seed, 'v_seed':v_seed}
                    quant_method[name].U = tffs[u_n].view(-1, u_n) @ Q_u.T
                    quant_method[name].V = tffs[v_n].view(-1, v_n) @ Q_v.T

                    # U = rand_ortho_butterfly(u_n).to(torch.float32)
                    # V = rand_ortho_butterfly(v_n).to(torch.float32)
                    # quant_method[name].U = tffs[u_n].view(-1, u_n) @ U.T
                    # quant_method[name].V = tffs[v_n].view(-1, v_n) @ V.T

                quant_method[name].name = name

            print(f' before compute H')
            logging.info(f' before compute H')

            def add_batch(name):

                def tmp(_, inp, out):
                    quant_method[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()
            # (H / nsamples).to(torch.float32)
            for name in subset:
                quant_method[name].post_batch()

            print(f'H computation Done')
            logging.info(f'H computation Done')

            for name in subset:
                if quant_method[name].nsamples == 0:
                    print(f'{name} has nsamples = 0')
                    breakpoint()
                    continue
                print(name)
                logging.info(name)

                print(f' layer i before preproc')
                logging.info(f' layer i before preproc')
                # command = "nvidia-smi --query-gpu=index,utilization.memory,memory.total,memory.used --format=csv --id=0,1"

                # if i == 8 and 'down' in name:
                #     os.environ['BKTPT'] = 'True'
                #     breakpoint()
                # else: 
                #     os.environ['BKTPT'] = 'False'

                quant_method[name].preproc(
                                    preproc_gptqH=args.pre_gptqH, percdamp=args.percdamp,
                                    preproc_rescale=args.pre_rescale, 
                                    preproc_proj=args.pre_proj, preproc_proj_extra=args.pre_proj_extra)
                print(f' layer i before fasterquant')
                logging.info(f' layer i before fasterquant')
                # coutput = subprocess.check_output(command, shell=True, text=True)
                # logging.info(coutput)
                if args.quant == 'gptq':
                    quant_method[name].fasterquant(groupsize=args.groupsize)
                elif args.quant in ['allbal','ldlq','ldlqRG','ldlbal_admm']:
                    quant_method[name].fasterquant(lazy_batch=args.lazy_batch)
                elif args.quant == 'nearest':
                    quant_method[name].fasterquant()

                # quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)
                quantizers['model.layers.%d.%s' % (i, name)] = quant_method[name].quantizer.cpu()

                logging.info(f'layer i after fasterquant')
                # coutput = subprocess.check_output(command, shell=True, text=True)
                # logging.info(coutput)

                if args.observe:
                    observer.submit(name=name, layerid=i, gptq=gptq[name], error=error)
                else:
                    errors.append(quant_method[name].error)
                    times.append(quant_method[name].time)
                    Hmags.append(quant_method[name].Hmag)
                    quant_method[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        
        print('after layer i, its still in GPU')
        logging.info('after layer i, its still in GPU')

        layers[i] = layer.module.cpu()

        print('after layer i, its in CPU')
        logging.info('after layer i, its in CPU')

        del layer
        del quant_method
        torch.cuda.empty_cache()

        print('after layer i, its deleted now')
        logging.info('after layer i, its deleted now')

        inps, outs = outs, inps
        print('+------------------+--------------+------------+-----------+-------+')
        print('\n')
        logging.info('+------------------+--------------+------------+-----------+-------+')
        logging.info('\n')

    if args.observe:
        observer.print()
        conditions = gen_conditions(args.wbits, args.groupsize)
        for item in observer.items():
            name = item[0]
            layerid = item[1]
            gptq = item[2]['gptq']
            error = item[2]['error']
            target = error / 2

            table = Texttable()
            table.header(['wbits', 'groupsize', 'error'])
            table.set_cols_dtype(['i', 'i', 'f'])
            table.add_row([args.wbits, args.groupsize, error])

            print('Optimizing {} {} ..'.format(name, layerid))
            for wbits, groupsize in conditions:

                if error < target:
                    # if error dropped 50%, skip
                    break

                gptq.quantizer.configure(wbits, perchannel=True, sym=args.sym, mse=False)

                scale, zero, g_idx, error = gptq.fasterquant(percdamp=args.percdamp, groupsize=groupsize, actorder=args.act_order, name=name)

                table.add_row([wbits, groupsize, error])
                quantizers['model.layers.%d.%s' % (layerid, name)] = (gptq.quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), wbits, groupsize)

            print(table.draw())
            print('\n')
            gptq.layer.to('cpu')
            gptq.free()

    model.config.use_cache = use_cache

    return quantizers

def get_llama(model):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = 2048
    return model

def load_quant(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    from transformers import LlamaConfig, LlamaForCausalLM, modeling_utils
    config = LlamaConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)

    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)
    model.seqlen = 2048
    print('Done.')

    return model

def llama_multigpu(model, gpus, gpu_dist):
    model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
    if hasattr(model.model, 'norm') and model.model.norm:
        model.model.norm = model.model.norm.to(gpus[0])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[0])

    cache = {'mask': None, 'position_ids': None}

    class MoveModule(nn.Module):

        def __init__(self, module, invalidate_cache):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
            self.invalidate_cache=invalidate_cache

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)

            if cache['mask'] is None or cache['mask'].device != self.dev or self.invalidate_cache:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']

            if cache['position_ids'] is None or cache['position_ids'].device != self.dev or self.invalidate_cache:
                cache['position_ids'] = kwargs['position_ids'].to(self.dev)
            kwargs['position_ids'] = cache['position_ids']
            
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.layers
    from math import ceil
    if not gpu_dist:
        pergpu = ceil(len(layers) / len(gpus))
        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(0 if i == 0 or i == len(layers) -1 else gpus[(i-1) // pergpu]), i==0)
    else:
        assert gpu_dist[0] >= 2, "At least two layers must be on GPU 0."
        assigned_gpus = [0] * (gpu_dist[0]-1)
        for i in range(1, len(gpu_dist)):
            assigned_gpus = assigned_gpus + [i] * gpu_dist[i]

        remaining_assignments = len(layers)-len(assigned_gpus) - 1
        if remaining_assignments > 0:
            assigned_gpus = assigned_gpus + [-1] * remaining_assignments

        assigned_gpus = assigned_gpus + [0]

        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(gpus[assigned_gpus[i]]), i==0)

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

    for i, layer in enumerate(model.model.layers):
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

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(input_ids[:, i:i + 1], past_key_values=cache['past'], attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1)))
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if hasattr(model, 'gpus'):
                mem_allocated = sum(torch.cuda.memory_allocated(gpu) for gpu in model.gpus) / 1024 / 1024
            else:
                mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory = max(max_memory, mem_allocated)
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())
            print('max memory(MiB):', max_memory)

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')
    logging.info('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
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
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = quant.Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())
    logging.info(ppl.item())

    model.config.use_cache = use_cache

    return ppl.item()

# TODO: perform packing on GPU
def llama_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    quant.make_quant_linear(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [quant.QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--exp_name", type=str, default='debug_thread',
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--img_size", type=int, default=224,
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
    parser.add_argument("--coef_est_type", type=str, default='weiner', choices = ['weiner', 'naive'],
                        help="how to estimate the weights from the quantized versions")
    parser.add_argument("--save_path", type=str, default=None, 
                        help="provide the savepath; otherwise a cat of exp_name and current time will be used")
    parser.add_argument("--parent_dir", type=str, default='test', 
                        help="parent dir for storing the results")
    parser.add_argument("--Weiner_m_diag_rank", type=int, default=3,
                        help="set the rank for the LowRank approximation of the residue after (Weiner - diag)")
    parser.add_argument('--tff_redundancy', type=float, default=1,
                        help="Redundancy in tffs")
    parser.add_argument('--emb_redundancy', type=float, default=1,
                        help="Redundancy in tffs for embedding quantization")
    parser.add_argument('--wiener_filt_en', action='store_true',
                        help="enable the Wiener filter after TFF based quantization")
    parser.add_argument('--clamp_noise_filt_en', action='store_true',
                        help="Wiener filter based clamping noise filter")
    parser.add_argument('--num_vals', type=int, default=1,
                        help="num vals")
    parser.add_argument('--x_sigma', type=float, default=2,
                        help="x times sigma for symm scale")
    parser.add_argument('--emb_quant_en', action='store_true',
                        help="enable quantization of embedding")

    parser.add_argument(
        '--percdamp',
        type=float,
        default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--quant',
                        choices=['allbal', 
                        'ldlq', 'ldlqRG', 'ldlbal_admm', 
                        'nearest', 'gptq'],
                        default='gptq',
                        help='Which quantization method to use.')
    parser.add_argument('--emb_quant',
                        choices=['ldlq', 
                        'ldlq', 'ldlqRG', 'ldlbal_admm', 
                        'nearest', 'gptq'],
                        default='nearest',
                        help='Which quantization method to use.')
    parser.add_argument('--emb_quant_only', action='store_true',
                        help='Return after quantizing embeddings')
    parser.add_argument(
        '--wbits',
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument(
        '--emb_wbits',
        type=int,
        default=16,
        choices=[2, 3, 4, 8, 16],
        help='#bits to use for embedding quantization; use 16 for evaluating base model.')
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
        '--emb_pre_gptqH',
        action='store_true',
        help='preprocessing')
    parser.add_argument(
        '--emb_pre_rescale',
        action='store_true',
        help='preprocessing')
    parser.add_argument(
        '--emb_pre_proj',
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
    parser.add_argument('--emb_qfn',
                        type=str,
                        default='a',
                        help='emb_qfn: a is default, b is sym incoherent based')
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


    parser.add_argument('model', type=str, help='llama model to load')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'], help='Where to extract calibration data from.')
    parser.add_argument('--nearest', action='store_true', help='Whether to run the RTN baseline.')
    parser.add_argument('--trits', action='store_true', help='Whether to use trits for quantization.')
    parser.add_argument('--eval', action='store_true', help='evaluate quantized model.')
    parser.add_argument('--test-generation', action='store_true', help='test generation.')
    parser.add_argument('--save_safetensors', type=str, default='', help='Save quantized `.safetensors` checkpoint under this name.')
    parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
    parser.add_argument('--new_eval', action='store_true', help='Whether to use the new PTB and C4 eval')
    parser.add_argument('--layers-dist', type=str, default='', help='Distribution of layers across GPUs. e.g. 2:1:1 for 2 layers on GPU 0, 1 layer on GPU 1, and 1 layer on GPU 2. Any remaining layers will be assigned to your last GPU.')
    parser.add_argument('--observe',
                        action='store_true',
                        help='Auto upgrade layer precision to higher precision, for example int2 to int4, groupsize 128 to 64. \
            When this feature enabled, `--save` or `--save_safetensors` would be disable.')
    parser.add_argument('--quant-directory', type=str, default=None, help='Specify the directory for export quantization parameters to toml format. `None` means no export by default.')

    args = parser.parse_args()

    import os
    # os.environ['BKTPT'] = 'False'

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
    # seed the experiment 
    set_seed(args)
    # torch.use_deterministic_algorithms(True)
    filename = f'logs/{args.exp_name}.log'
    logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # results_dir = 'output_sigma'
    results_dir = 'output_llama'

    if args.layers_dist:
        gpu_dist = [int(x) for x in args.layers_dist.split(':')]
    else:
        gpu_dist = []

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
    else:
        model = get_llama(args.model)
        model.eval()

    dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen)

    if not args.load and args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV, args.seed)
        print(time.time() - tick)

    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark(model, input_ids, check=args.check)

    if args.exp_name != 'debug_thread':
        #     opt_pack3(model, quantizers)
        # save the model
        write_path = os.path.join(results_dir, f'{args.parent_dir}', f'wb{args.wbits}')
        os.makedirs(write_path, exist_ok=True)
        model_dir = os.path.join(write_path, args.exp_name)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, 'Qmodel.pth'))
        
        try:
            mdict = {   'quantizers': quantizers,
                    }
            torch.save(mdict, os.path.join(model_dir, 'Qparams.pth'))
        except:
            pass

    if args.eval:
        print(f'evaluating')
        ppls = []
        datasets = ['wikitext2', 'ptb', 'c4']
        if args.new_eval:
            datasets = ['wikitext2', 'ptb-new', 'c4-new']
        for dataset in datasets:
            print(dataset)
            dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
            print(dataset)
            logging.info(dataset)
            ppl = llama_eval(model, testloader, DEV)
            ppls.append(ppl)
    
        logging.info('------------------------------------------------------------------------')
        logging.info('------------------------------------------------------------------------')

        import csv
        import os
        results  = [args.exp_name, args.tff_redundancy, args.emb_redundancy, args.emb_quant]
        results.extend(ppls)
        csv_file_path = os.path.join(write_path,'results.csv')
        with open(csv_file_path, mode='a', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(results)
    
    if args.test_generation:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)

        from transformers import LlamaTokenizer, TextStreamer
        tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
        input_ids = tokenizer(["The capital of New Mexico is"], return_tensors="pt").input_ids.to(gpus[0])
        streamer = TextStreamer(tokenizer)
        with torch.no_grad():
            generated_ids = model.generate(input_ids, streamer=streamer)
        
    if args.quant_directory is not None:
        export_quant_table(quantizers, args.quant_directory)

    if not args.observe and args.save:
        llama_pack(model, quantizers, args.wbits, args.groupsize)
        torch.save(model.state_dict(), args.save)

    if not args.observe and args.save_safetensors:
        llama_pack(model, quantizers, args.wbits, args.groupsize)
        from safetensors.torch import save_file as safe_save
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        safe_save(state_dict, args.save_safetensors)
