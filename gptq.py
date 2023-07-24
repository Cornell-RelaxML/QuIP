import math
import time

import torch
import torch.nn as nn
import transformers

from method import QuantMethod
from quant import *

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ(QuantMethod):

    def fasterquant(self, blocksize=128, groupsize=-1, copy_H=False, debug_equiv=False):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        # when check LDL and GPTQ equiv, need float64 for numerics
        if not debug_equiv:
            W = W.float()
        full_W = W.clone()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        # for saving H
        if copy_H:
            H = self.H.data.clone()
        else:
            H = self.H
        # note: removed and put in QuantMethod
        # dead = torch.diag(H) == 0
        # H[dead, dead] = 1
        # W[:, dead] = 0
        # damp = percdamp * torch.mean(torch.diag(H))
        # diag = torch.arange(self.columns, device=self.dev)
        # H[diag, diag] += damp

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i +
                                                                  groupsize)],
                                                   weight=True)

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                # q = quantize(w.unsqueeze(1), self.quantizer.scale,
                #              self.quantizer.zero,
                #              self.quantizer.maxq).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q)**2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1)**2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        # print('time %.2f' % (time.time() - tick))
        # print('error', torch.sum(Losses).item())
        self.time = time.time() - tick

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype)

        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1)**2))

        self.postproc()
        self.error_compute(full_W, self.layer.weight.data)
        # to preserve H for saveH
        if not copy_H:
            del self.H

