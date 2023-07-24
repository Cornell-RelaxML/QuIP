import time
import torch

import torch
import torch.nn as nn
import transformers

#from gptq import GPTQ
from method import QuantMethod
from vector_balance import quantize_weight_vecbal 


class Balance(QuantMethod):

    def configure(self, qmethod, nbits, npasses, unbiased):
        self.qmethod = qmethod
        self.nbits = nbits
        self.npasses = npasses
        self.unbiased = unbiased

    def fasterquant(self, lazy_batch=False):
        w = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            raise NotImplementedError()
        if isinstance(self.layer, transformers.Conv1D):
            raise NotImplementedError()
        tick = time.time()
        if not self.quantizer.ready():
            self.quantizer.find_params(w, weight=True)
        H = self.H.data.clone()

        quant_w = quantize_weight_vecbal(
            w=w, H=H,
            nbits=self.nbits,
            npasses=self.npasses,
            scale=self.quantizer.scale,
            zero=self.quantizer.zero,
            maxq=self.quantizer.maxq,
            unbiased=self.unbiased,
            qfn=self.quantizer.qfn,
            qmethod=self.qmethod,
            lazy_batch=lazy_batch
        )
        self.layer.weight.data = quant_w
        self.postproc()
        # print('time %.2f' % (time.time() - tick))
        self.time = time.time() - tick
        self.error_compute(w, quant_w)

