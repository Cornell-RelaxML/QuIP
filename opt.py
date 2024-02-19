import time

import torch
import torch.nn as nn

from gptq import *
from bal import Balance
from near import Nearest
from modelutils import *
from quant import *

from tqdm import tqdm

from utils.construct_tff import construct_real_tff
import logging
from typing import List, Optional, Tuple, Union

import os

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
def opt_emb_quant(subset, dev, args):
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
            k_tff = round(u_n // l_tff * args.tff_redundancy)
            tffs_u = construct_real_tff(k_tff, l_tff // 2, u_n // 2)

            l_tff = v_n // 16
            k_tff = round(v_n // l_tff * args.tff_redundancy)
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

    # Do the embed quantization
    if args.emb_quant_en:
        name = 'emb'
        subset = {name: model.model.decoder.embed_tokens}
        opt_emb_quant(subset, dev, args)

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

    tffs = {}

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')


    quantizers = {}
    tff_rand_seeds = {}
    quantized_weights = {}
    Wiener_params = {}
    errors, Hmags, times = [], [], []
    for i in tqdm(range(len(layers))):

        class dual_gpu_layer(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                # send attention to cuda 0, mlp to cuda 1
                self.module.self_attn               = self.module.self_attn.to('cuda:0')
                self.module.self_attn_layer_norm    = self.module.self_attn_layer_norm.to('cuda:0')

                self.module.activation_fn = self.module.activation_fn.to('cuda:1')
                self.module.fc1 = self.module.fc1.to('cuda:1')
                self.module.fc2 = self.module.fc2.to('cuda:1')
                self.module.final_layer_norm = self.module.final_layer_norm.to('cuda:1')

            def forward(
                self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                layer_head_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: Optional[bool] = False,
                use_cache: Optional[bool] = False,
            ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

                # Device 0
                hidden_states = hidden_states.to('cuda:0')
                residual = hidden_states

                # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
                if self.module.do_layer_norm_before:
                    hidden_states = self.module.self_attn_layer_norm(hidden_states)

                # Self Attention
                hidden_states, self_attn_weights, present_key_value = self.module.self_attn(
                    hidden_states=hidden_states,
                    past_key_value=past_key_value,
                    attention_mask=attention_mask,
                    layer_head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                )
                hidden_states = nn.functional.dropout(hidden_states, p=self.module.dropout, training=self.module.training)
                hidden_states = residual + hidden_states

                # 350m applies layer norm AFTER attention
                if not self.module.do_layer_norm_before:
                    hidden_states = self.module.self_attn_layer_norm(hidden_states)

                # Device 1
                # Fully Connected
                hidden_states = hidden_states.to('cuda:1')
                hidden_states_shape = hidden_states.shape
                hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
                residual = hidden_states

                # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
                if self.module.do_layer_norm_before:
                    hidden_states = self.module.final_layer_norm(hidden_states)

                hidden_states = self.module.fc1(hidden_states)
                hidden_states = self.module.activation_fn(hidden_states)

                hidden_states = self.module.fc2(hidden_states)
                hidden_states = nn.functional.dropout(hidden_states, p=self.module.dropout, training=self.module.training)

                hidden_states = (residual + hidden_states).view(hidden_states_shape)

                # 350m applies layer norm AFTER attention
                if not self.module.do_layer_norm_before:
                    hidden_states = self.module.final_layer_norm(hidden_states)

                outputs = (hidden_states,)

                if output_attentions:
                    outputs += (self_attn_weights,)

                if use_cache:
                    outputs += (present_key_value,)

                return outputs


        layer = dual_gpu_layer(layers[i])

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
                                               x_sigma= args.x_sigma)
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
                                               x_sigma= args.x_sigma)

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
                u_seed = g_u.seed()
                rand_mat_u = torch.randn((u_n, u_n), generator=g_u)
                Q_u, _ = torch.linalg.qr(rand_mat_u)
                g_v = torch.Generator() # use this to store the seed for later
                v_seed = g_v.seed()
                rand_mat_v = torch.randn((v_n, v_n), generator=g_v)
                Q_v, _ = torch.linalg.qr(rand_mat_v)
                tff_rand_seeds[f'quantized model.decoder.layers.{i}.{name}'] = {'u_seed': u_seed, 'v_seed':v_seed}
                quant_method[name].U = tffs[u_n].view(-1, u_n) @ Q_u.T
                quant_method[name].V = tffs[v_n].view(-1, v_n) @ Q_v.T

            quant_method[name].name = name

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
                        (i, name)] = quant_method[name].quantizer.to('cpu')

            errors.append(quant_method[name].error)
            times.append(quant_method[name].time)
            Hmags.append(quant_method[name].Hmag)
            quant_method[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0),
                            attention_mask=attention_mask)[0]

        layers[i] = layer.module.cpu()
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
    logging.info(ppl.item())

    model.config.use_cache = use_cache

    return ppl.item()


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
    # from datautils import *
    from utils.llama_utils import get_loaders

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
    parser.add_argument("--parent_dir", type=str, default='temp', 
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
    parser.add_argument('--x_sigma', type=float, default=2,
                        help="x times sigma for symm scale")
    parser.add_argument('--emb_quant_en', action='store_true',
                        help="enable quantization of embedding")


    parser.add_argument('model',
                        type=str,
                        help='OPT model to load; pass `facebook/opt-X`.')
    parser.add_argument('dataset',
                        type=str,
                        choices=['wikitext2', 'ptb', 'c4'],
                        help='Where to extract calibration data from.')
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
    parser.add_argument('--emb_quant',
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
        '--emb_wbits',
        type=int,
        default=16,
        choices=[2, 3, 4, 16],
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
                        help='qfn: a is default, b is sym incoherent based')
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

    args = parser.parse_args()
    # defaults to incoherence processing
    if args.incoh_processing:
        args.pre_gptqH   = True
        args.pre_rescale = True
        args.pre_proj    = True
        args.proj_extra  = 1
        args.qfn         = 'b'

    results_dir = 'output_opt'
    # results_dir = '/nobackup/harsha/storage/framequant/output_opt'
    log_dir = os.path.join(results_dir, f'logs')
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, f'{args.exp_name}.log')
    logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.load:
        model = load_quant(args.model, args.load)
        model.eval()
    else:
        model = get_opt(args.model)
        model.eval()

        dataloader, _ = get_loaders(        args.dataset,
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
        ppls = []
        for dataset in ['wikitext2', 'ptb-new', 'c4-new']:
        # for dataset in ['wikitext2', 'ptb-new']:
            dataloader, testloader = get_loaders(dataset,
                                                seed=args.seed,
                                                model=args.model,
                                                seqlen=model.seqlen)
            print(dataset)
            logging.info(dataset)
            ppl = opt_eval(model, testloader, DEV)
            ppls.append(ppl)
    logging.info('------------------------------------------------------------------------')
    logging.info('------------------------------------------------------------------------')

    import csv
    import os
    results  = [args.exp_name, args.tff_redundancy, args.emb_quant]
    results.extend(ppls)
    write_path = os.path.join(results_dir, f'{args.parent_dir}', f'wb{args.wbits}')
    os.makedirs(write_path, exist_ok=True)
    csv_file_path = os.path.join(write_path,'results.csv')
    with open(csv_file_path, mode='a', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(results)
