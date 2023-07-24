import math
import time

import torch
import torch.nn as nn
import transformers

import primefac
import scipy
import math

from quant import Quantizer

DEBUG = False

def butterfly_factors(n):
    pf = list(primefac.primefac(n))
    return (math.prod(pf[0::2]), math.prod(pf[1::2]))

def gen_rand_orthos(m,p):
    if (p != 2):
        return torch.tensor(scipy.stats.special_ortho_group.rvs(p, size=m)).to(torch.float32)
    X = torch.zeros(m,2,2)
    t = torch.rand(m) * (2 * math.pi) 
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    X[:,0,0] = cos_t
    X[:,1,1] = cos_t
    X[:,0,1] = sin_t
    X[:,1,0] = -sin_t
    return X

# generates a random orthogonal butterfly matrix of dimension n
def gen_rand_ortho_butterfly(n):
    return ([gen_rand_orthos(n//p, p) for p in butterfly_factors(n)], torch.randperm(n), torch.randperm(n))

# generates a random orthogonal butterfly matrix of dimension n, without blocking
def gen_rand_ortho_butterfly_noblock(n):
    return ([gen_rand_orthos(1, p) for p in butterfly_factors(n)], torch.randperm(n), torch.randperm(n))

# generates a random orthogonal butterfly matrix of dimension n, no permutation, but yes blocking
def gen_rand_ortho_butterfly_nopermute(n):
    return ([gen_rand_orthos(n//p, p) for p in butterfly_factors(n)], torch.arange(n), torch.arange(n))

# multiply by a random orthogonal butterfly matrix
def mul_ortho_butterfly(Bpp, x):
    (B, p_in, p_out) = Bpp
    assert((len(x.shape) == 1) or (len(x.shape) == 2))
    orig_dim = 2
    if (len(x.shape) == 1):
        (n,) = x.shape
        x = x.reshape(n,1)
        orig_dim = 1
    (n,q) = x.shape
    x = x[p_in,:]
    pfn = tuple(butterfly_factors(n))
    for i in range(len(pfn)):
        mpfx = math.prod(pfn[0:i])
        p = pfn[i]
        msfx = math.prod(pfn[(i+1):])
        x = x.reshape(mpfx, p, msfx, q).permute(0,2,1,3).reshape(mpfx * msfx, p, q)
        x = B[i] @ x
        x = x.reshape(mpfx, msfx, p, q).permute(0,2,1,3).reshape(n,q)
    x = x[p_out,:]
    if (orig_dim == 1):
        x = x.reshape(n)
    return x

# generates a random orthogonal butterfly matrix of dimension n
# and converts it to a dense matrix
def rand_ortho_butterfly(n):
    return mul_ortho_butterfly(gen_rand_ortho_butterfly(n), torch.eye(n))

def rand_ortho_butterfly_noblock(n):
    return mul_ortho_butterfly(gen_rand_ortho_butterfly_noblock(n), torch.eye(n))

def rand_ortho_butterfly_nopermute(n):
    return mul_ortho_butterfly(gen_rand_ortho_butterfly_nopermute(n), torch.eye(n))

class QuantMethod:
    '''
    Base class for quantization methods
    '''
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), dtype=torch.float64, device=self.dev)
        self.nsamples = 0
        self.preproc_done = False

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
                self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(self.layer.kernel_size,
                               dilation=self.layer.dilation,
                               padding=self.layer.padding,
                               stride=self.layer.stride)
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.nsamples += tmp
        inp = inp.to(torch.float64)
        self.H.add_(inp.matmul(inp.t()))

    def post_batch(self):
        self.H = (self.H / self.nsamples).to(torch.float32)

    def preproc(self, preproc_gptqH=False, percdamp=.01,
                preproc_rescale=False, preproc_proj=False, preproc_proj_extra=0):
        """
        optional preprocessing: scales w,H diagonally, or random projection
        run gptqH last
        preproc_proj_extra:
        0: 2 factor butterfly + permute
        1: 2 factor butterfly + permute + no blocking
        2: 2 factor butterfly + no permute
        3: random orthogonal
        """
        self.preproc_gptqH   = preproc_gptqH
        self.preproc_rescale = preproc_rescale
        self.preproc_proj    = preproc_proj
        if preproc_rescale:
            w = self.layer.weight.data.clone().to(torch.float32)
            H = self.H.to(torch.float32)
            H /= H.abs().max()
            diagH = torch.diag(H)
            diagW2 = torch.diag(w.T @ w)
            diagH = torch.clamp(diagH, min=1e-8)
            diagW2 = torch.clamp(diagW2, min=1e-8)
            scaleWH = (diagH / diagW2).sqrt().sqrt().to(torch.float32)
            scaleWH = scaleWH.clamp(min=1e-8)
            w *= scaleWH[None,:]
            H /= scaleWH[None,:]
            H /= scaleWH[:,None]
            w = w.to(torch.float32)
            scaleWH = scaleWH.to(torch.float32)
            self.scaleWH = scaleWH.cpu()
            self.layer.weight.data = w.to(self.layer.weight.data.dtype)
            self.H.data = H.to(self.H.data.dtype)
        if preproc_proj:
            w = self.layer.weight.data.clone().to(torch.float32)
            H = self.H.data.clone().to(torch.float32)
            # 
            if preproc_proj_extra == 0:
                U = rand_ortho_butterfly(w.shape[0]).to(torch.float32).to(w.device)
                V = rand_ortho_butterfly(w.shape[1]).to(torch.float32).to(w.device)
            elif preproc_proj_extra == 1:
                U = rand_ortho_butterfly_noblock(w.shape[0]).to(torch.float32).to(w.device)
                V = rand_ortho_butterfly_noblock(w.shape[1]).to(torch.float32).to(w.device)
            elif preproc_proj_extra == 2:
                U = rand_ortho_butterfly_nopermute(w.shape[0]).to(torch.float32).to(w.device)
                V = rand_ortho_butterfly_nopermute(w.shape[1]).to(torch.float32).to(w.device)
            #EH = torch.linalg.eigh(H)
            #H = (EH.eigenvectors @ torch.diag(EH.eigenvalues.relu() * H.shape[0] / (EH.eigenvalues.relu().sum() + 1e-8) + 1e-2) @ EH.eigenvectors.T).to(w.device)
            #H = H.to(torch.float32)
            H = H * (H.shape[0] / (torch.trace(H) + 1e-8)) + 1e-2 * torch.eye(H.shape[0], device=w.device)
            H = H.to(torch.float32)
            w = U @ w @ V.T
            H = V @ H @ V.T
            self.projU = U.cpu()
            self.projV = V.cpu()
            self.layer.weight.data = w.to(self.layer.weight.data.dtype)
            self.H.data = H.to(self.H.data.dtype)
        # H modification from gptq
        if self.preproc_gptqH:
            w = self.layer.weight.data.clone()
            H = self.H.data.clone()
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            w[:, dead] = 0
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.dev)
            H[diag, diag] += damp
            self.layer.weight.data = w.to(self.layer.weight.data.dtype)
            self.H.data = H.to(self.H.data.dtype)
        self.preproc_done = True
    
    def postproc(self):
        assert self.preproc_done is True
        if self.preproc_proj:
            w = self.layer.weight.data.clone().to(torch.float32)
            H = self.H.data.clone().to(torch.float32)
            U = self.projU.to(w.device)
            V = self.projV.to(w.device)
            w = (U.T @ w @ V)
            H = (V.T @ H @ V)
            self.layer.weight.data = w.to(self.layer.weight.data.dtype)
            self.H.data = H.to(self.H.data.dtype)
        if self.preproc_rescale:
            w = self.layer.weight.data.clone()
            H = self.H.data.clone()
            scaleWH = self.scaleWH.to(w.device)
            w = w / scaleWH[None,:]
            H = H * scaleWH[:,None]
            H = H * scaleWH[None,:]
            self.layer.weight.data = w.to(self.layer.weight.data.dtype)
            self.H.data = H.to(self.H.data.dtype)

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H        = None
        self.Losses   = None
        self.Trace    = None
        self.scaleWH  = None
        self.projU    = None
        self.projV    = None
        torch.cuda.empty_cache()

    def error_compute(self, full_W, quant_W):
        full_W = full_W.float()
        quant_W = quant_W.float()
        self.error = ((full_W - quant_W) @ self.H.type(torch.float)
                      @ (full_W - quant_W).T).trace().item()
        self.Hmag = self.H.max().item()