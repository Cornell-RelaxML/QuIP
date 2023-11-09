import torch
import math

# Tight Fusion Frame Existence Test
def tffet(k,l,n):
    if 2*l > n:
        l = n-l
    exists = 'unknown'
    while exists == 'unknown':
        if n%l == 0:
            if k >= n/l:
                exists = True
            else:
                exists = False
        else:
            if k > math.ceil(n/l) + 1:
                exists = True
            elif k < math.ceil(n/l) + 1:
                exists = False
            else:
                n = k*l - n
                l = n-l

    return exists

def insert_tx(x, tff, r,c):
    tx = (1/math.sqrt(2)) * torch.as_tensor([[math.sqrt(x), math.sqrt(x)], [math.sqrt(2-x), -math.sqrt(2-x)]])
    tff[r:r+2, c:c+2] = tx

# construct tight fusion frames
def construct_tight_frames(k,l,n):
    existence = tffet(k,l,n)
    if not existence:
        print('the given k,l,n values are invalid')
        exit

    # constructing a n length frame for C^l
    tff_lxn = torch.zeros((l,n))
    tff_lxn[0,0] = 0
    tff_lxn[0,1] = 0
    target_norm = n/l

    # fill the matrix
    col = 0
    for row in range(l):
        curr_norm = torch.norm(tff_lxn[row, :col])**2
        req_norm = target_norm - curr_norm
        while (req_norm >=1.) or math.isclose(req_norm, 1,abs_tol = 1e-5):
            tff_lxn[row, col] = 1
            req_norm -= 1
            col += 1
        if not math.isclose(req_norm, 0, abs_tol=1e-5):
            insert_tx(req_norm, tff_lxn, row, col)
            col += 2

    # part 2 - Modulated TFFs
    tffs = []
    for _k in range(k):
        wt_mat = torch.as_tensor([2*math.pi*(_k)*_n/k for _n in range(n)])
        tffs.append(torch.polar(tff_lxn, wt_mat))
    # tffs = torch.stack(tffs,0) * math.sqrt(l/n)
    tffs = torch.stack(tffs,0) * math.sqrt(1/k)
    return tffs

def construct_real_tff(k,l,n):
    tffs = construct_tight_frames(k,l,n) # / math.sqrt(2) # normalize the complex basis to get unit norm

    # size of real tffs is (k,2l,2n)
    tffs_intrmd = torch.view_as_real(tffs)
    tffs_even_l = tffs_intrmd.view(k,l,2*n) * ((-1)**torch.arange(2*n))
    tffs_odd_l = torch.roll(tffs_intrmd, shifts=(1), dims=(-1)).view(k,l,2*n)
    tffs_real = torch.stack((tffs_even_l, tffs_odd_l), dim=2).view(k,2*l,2*n)

    return tffs_real

if __name__ == "__main__":
    # k = 10
    # l = 3
    # n = 20
    # k = 5
    # l = 4
    # n = 11
    # # construct TFFs
    # tffs = construct_tight_frames(k,l,n)

    k = 256
    l = 96
    n = 768
    tffs = construct_real_tff(k,l,n)

