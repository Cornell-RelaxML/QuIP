import glob
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import pickle

# loop over all preproc1 H:
# tr(D) / tr(H)
# matrix rank of H
# compute ||eigenvector||_1 / sqrt{n}

# loop over all preproc2 H:
# tr(D) / tr(H)
# matrix rank of H
# compute ||eigenvector||_1 / sqrt{n}

def Hsummary(H, percdamp=0.01):
    assert H.shape[0] == H.shape[1]
    n = H.shape[0]
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(n)
    H[diag, diag] += damp
    L = torch.linalg.cholesky(H)
    D = torch.diag(L).square()
    a = D.sum() / H.trace()
    k00 = torch.linalg.matrix_rank(H) / n
    k01 = torch.linalg.matrix_rank(H, rtol=0.01) / n
    _, Q = torch.linalg.eigh(H)
    mu = torch.linalg.matrix_norm(Q) * np.sqrt(n)
    return a, k00, k01, mu

def collect(dirname, savename):
    a_ls, k00_ls, k01_ls, mu_ls = [], [], [], []
    for fname in tqdm(glob.glob(dirname+'/*.pt')):
        H = torch.load(fname)
        print(f"{fname}, H.shape: {H.shape}")
        a, k00, k01, mu = Hsummary(H)
        a_ls.append(a)
        k00_ls.append(k00)
        k01_ls.append(k01)
        mu_ls.append(mu)
    a_ls = np.array(a_ls)
    k00_ls = np.array(k00_ls)
    k01_ls = np.array(k01_ls)
    mu_ls = np.array(mu_ls)
    print(f"tr(D) / tr(H): {np.mean(a_ls)} (+/- {np.std(a_ls)})")
    print(f"matrix rank rtol=0.00: {np.mean(k00_ls)} (+/- {np.std(k00_ls)})")
    print(f"matrix rank rtol=0.01: {np.mean(k01_ls)} (+/- {np.std(k01_ls)})")
    print(f"incoherency mu: {np.mean(mu_ls)} (+/- {np.std(mu_ls)})")
    with open(savename, 'wb') as f:
        pickle.dump({
            'trDtrH': a_ls,
            'rank_rtol0': k00_ls,
            'rank_rtol01': k01_ls,
            'incoh_mu': mu_ls
        }, f)

p1 = [
    "slurm/H_run2/opt-125m_gptq_W4_preproc1",
    "slurm/H_run2/opt-350m_gptq_W4_preproc1",
    "slurm/H_run2/opt-1.3b_gptq_W4_preproc1",
    "slurm/H_run2/opt-2.7b_gptq_W4_preproc1",
]
p2 = [
    "slurm/H_opt-125m_run1/opt-125m_gptq_W4_preproc2",
    "slurm/H_opt-350m_run1/opt-350m_gptq_W4_preproc2",
    "slurm/H_opt-1.3b_run1/opt-1.3b_gptq_W4_preproc2",
    "slurm/H_opt-2.7b_run1/opt-2.7b_gptq_W4_preproc2",
]

def save_spectrum(fname, savename):
    """ slurm/Hspectrum/...
    """
    H = torch.load(fname)
    n = H.shape[0]
    percdamp = 0.01
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(n)
    H[diag, diag] += damp 
    L = torch.linalg.eigvalsh(H).numpy()
    L = pd.DataFrame(L)
    L.to_csv(savename)

def kick_spectrum():
    # save_spectrum(
    #     "slurm/H_run2/opt-2.7b_gptq_W4_preproc1/H_model.decoder.layers.8.fc2.pt",
    #     "slurm/Hspectrum/opt-2.7b_8fc2_preproc1.csv"
    #     )
    # save_spectrum(
    #     "slurm/H_run2/opt-2.7b_gptq_W4_preproc1/H_model.decoder.layers.20.self_attn.q_proj.pt",
    #     "slurm/Hspectrum/opt-2.7b_20qproj_preproc1.csv"
    #     )
    # save_spectrum(
    #     "slurm/H_opt-2.7b_run1/opt-2.7b_gptq_W4_preproc2/H_model.decoder.layers.8.fc2.pt",
    #     "slurm/Hspectrum/opt-2.7b_8fc2_preproc2.csv"
    #     )
    # save_spectrum(
    #     "slurm/H_opt-2.7b_run1/opt-2.7b_gptq_W4_preproc2/H_model.decoder.layers.20.self_attn.q_proj.pt",
    #     "slurm/Hspectrum/opt-2.7b_20qproj_preproc2.csv"
    #     )
    save_spectrum(
        "slurm/H_run2/opt-2.7b_gptq_W4_preproc1/H_model.decoder.layers.16.self_attn.k_proj.pt",
        "slurm/Hspectrum/opt-2.7b_16kproj_preproc1.csv"
        )
    save_spectrum(
        "slurm/H_run2/opt-2.7b_gptq_W4_preproc1/H_model.decoder.layers.30.fc1.pt",
        "slurm/Hspectrum/opt-2.7b_30fc1_preproc1.csv"
        )
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname',
                        type=str)
    parser.add_argument('--savename',
                        type=str)
    args = parser.parse_args()

    collect(args.dirname, args.savename)