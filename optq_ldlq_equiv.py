import copy
import time
import torch
from quant import Quantizer
from gptq import GPTQ
from bal import Balance
from near import Nearest

class FakeLayer():
    def __init__(self, m=10, d=10):
        self.weight = torch.rand(m, d, dtype=torch.float64)
        x = torch.rand(d,d, dtype=torch.float64)
        self.H = x.T @ x  + 0.01 * torch.eye(d)

if __name__ == "__main__":
    wbits = 3
    layer = FakeLayer(m=1000, d=1000)
    layer_gptq = copy.deepcopy(layer)
    layer_ldl  = copy.deepcopy(layer)
    layer_near  = copy.deepcopy(layer)

    # initialize GPTQ, LDL, & Nearest
    gptq_method = GPTQ(layer_gptq)
    gptq_method.H = layer_gptq.H
    gptq_method.quantizer = Quantizer()
    gptq_method.quantizer.configure(wbits,
                                    perchannel=True,
                                    sym=False,
                                    qfn='c', #LDLQ and GPTQ, round in same order
                                    mse=False)
    gptq_method.preproc(preproc_gptqH=False, percdamp=0,
                        preproc_rescale=False, 
                        preproc_proj=False, preproc_proj_extra=0)
    ldl_method = Balance(layer_ldl)
    ldl_method.H = layer_ldl.H
    ldl_method.configure('ldl_gptqequiv',
                         wbits, 
                         npasses=1,
                         unbiased=False)
    ldl_method.quantizer = Quantizer()
    ldl_method.quantizer.configure(wbits,
                                   perchannel=True,
                                   sym=False,
                                   qfn='a',
                                   mse=False)
    ldl_method.preproc(preproc_gptqH=False, percdamp=0,
                        preproc_rescale=False, 
                        preproc_proj=False, preproc_proj_extra=0)
    near_method = Nearest(layer_near)
    near_method.H = layer_near.H
    near_method.quantizer = Quantizer()
    near_method.quantizer.configure(wbits,
                                   perchannel=True,
                                   sym=False,
                                   qfn='a',
                                   mse=False)
    near_method.preproc(preproc_gptqH=False, percdamp=0,
                        preproc_rescale=False, 
                        preproc_proj=False, preproc_proj_extra=0)


    # Quantize
    t0 = time.time()
    gptq_method.fasterquant(groupsize=-1, debug_equiv=True)
    t1 = time.time()
    ldl_method.fasterquant()
    t2 = time.time()
    near_method.fasterquant()
    t3 = time.time()

    assert all(gptq_method.quantizer.scale == ldl_method.quantizer.scale)
    assert all(gptq_method.quantizer.zero == ldl_method.quantizer.zero)
    assert gptq_method.quantizer.maxq == ldl_method.quantizer.maxq

    print('=========== OPTQ vs. LDL Quant Weight Error ==================')
    err = (layer_gptq.weight - layer_ldl.weight).abs()
    err[layer_gptq.weight == layer_ldl.weight] = 0
    print(f"max elementwise difference: {err.max():.3f}")
    print(f"avg elementwise difference: {err.mean():.3f} (+/-{err.std():.3f})")

    print(f"frac less than 1e-6 apart: {torch.sum((layer_gptq.weight - layer_ldl.weight).abs() < 1e-6) / (layer.H.shape[0]*layer.H.shape[1]):.3f}")
    print(f"frac less than 1e-3 apart: {torch.sum((layer_gptq.weight - layer_ldl.weight).abs() < 1e-3) / (layer.H.shape[0]*layer.H.shape[1]):.3f}")

    print('=========== OPTQ vs. Nearest Quant Weight Error ==================')
    err = (layer_gptq.weight - layer_near.weight).abs()
    err[layer_gptq.weight == layer_near.weight] = 0
    print(f"max elementwise difference: {err.max():.3f}")
    print(f"avg elementwise difference: {err.mean():.3f} (+/-{err.std():.3f})")

    print(f"frac less than 1e-6 apart: {torch.sum((layer_gptq.weight - layer_near.weight).abs() < 1e-6) / (layer.H.shape[0]*layer.H.shape[1]):.3f}")
    print(f"frac less than 1e-3 apart: {torch.sum((layer_gptq.weight - layer_near.weight).abs() < 1e-3) / (layer.H.shape[0]*layer.H.shape[1]):.3f}")

    print('=========== Time and Proxy ==================')

    print(f"gptq took {t1-t0:.3f} seconds, ldl took {t2-t1:.3f} seconds, near took {t3-t2:.3f} seconds")
    print(f"gptq proxy is {gptq_method.error:.3e}, ldl proxy is {ldl_method.error:.3e}, near proxy is {near_method.error:.3e}")