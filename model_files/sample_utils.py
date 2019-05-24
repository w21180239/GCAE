import numpy as np

def compute_probs(losses):
    total_losses = np.sum(losses)
    probs = losses / total_losses
    return probs

def init_sampleTabl(N, probs):
    M = 0
    N = 5*N
    sample_tabl = np.zeros((N)).astype(int)
    # print(probs)
    Npi = N * probs
    # print(Npi)
    fl_Npi = np.floor(Npi).astype(int)
    for id, n in enumerate(fl_Npi):
        sample_tabl[M:M + n] = id
        M += n
    if M < N:
        res_Npi = fl_Npi - Npi
        ids = np.argsort(- res_Npi)
        # print(N)
        # print(M)
        # print(len(ids))
        sample_tabl[M:] = ids[:N - M]
    return sample_tabl

def sample(probs):
    N, M = len(probs), 0
    sample_tabl = init_sampleTabl(N, probs)
    sampled_ids = np.random.choice(sample_tabl, N)
    return sampled_ids

def subsample(losses):
    probs = compute_probs(losses)
    sampled_ids = sample(probs)
    return probs, sampled_ids