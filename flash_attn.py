import torch

def Flash_Attention(Q, K, V, block_size):
    N, d = Q.shape

    O = torch.zeros_like(Q)

    for i in range(0, N, block_size):
        Q_i = Q[i:i+block_size]

        m = torch.full((Q_i.shape[0], ), float('-inf'), device=Q.device)
        l = torch.zeros((Q_i.shape[0], ), device=Q.device)
        o = torch.zeros_like(Q_i)

        for j in range(0, N, block_size):
            K_j = K[j:j+block_size]
            V_j = V[j:j+block_size]

            S = Q_i @ K_j.T / (d ** 0.5)

            m_new = torch.maximum(m, torch.max(S, dim=1).values)  # shape (block_size, )
            
            exp_old = torch.exp(m-m_new) * l
            exp_new = torch.exp(S-m_new[:, None]).sum(dim=1)  # shape (block_size, )

            l_new = exp_old + exp_new
            o_new = (o * exp_old[:, None] + (torch.exp(S-m_new[:, None]) @ V_j)) / l_new[:, None]

            m = m_new
            l = l_new
            o = o_new

        O[i:i+block_size] = o

    return O
