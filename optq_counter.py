import torch
from vector_balance import round_ldl_gptqequiv

def proxy_loss(dw, H):
    return (dw @ H @ dw.T).trace()

def counter(n, d, c):
    H = torch.ones(n,n) + torch.eye(n)
    H[n-1,n-1] = 1.0
    H[0,1:(n-1)] += 2 * c
    H[1:(n-1),0] += 2 * c
    H[0,n-1] += c
    H[n-1,0] += c
    H[0,0] += 4 * c + n * (c**2)
    w = 0.499 * torch.ones(d,n) + 0.002 * (torch.arange(n) % 2)

    # quantize
    w_ldl = round_ldl_gptqequiv(w, H, 2)
    w_ldl_stoch = round_ldl_gptqequiv(w, H, 2, unbiased=True)
    w_near = w.round()
    w_stoch = (w + (torch.rand(d,n))).floor()

    ldl_loss = proxy_loss(w_ldl-w, H)
    ldl_stoch_loss = proxy_loss(w_ldl_stoch-w, H)
    near_loss = proxy_loss(w_near-w, H)
    stoch_loss = proxy_loss(w_stoch-w, H)

    print(f"ldl_loss: {ldl_loss}")
    print(f"ldl_stoch_loss: {ldl_stoch_loss}")
    print(f"near_loss: {near_loss}")
    print(f"stoch_loss: {stoch_loss}")

if __name__ == "__main__":
    # counter(128, 4, 0.01)
    counter(256, 256, 0.01)
    print('-----')
    counter(512, 512, 0.01)
    print('-----')
    counter(1024, 1024, 0.01)
    print('-----')
    counter(2048, 2048, 0.01)
    print('-----')
    counter(4096, 4096, 0.01)