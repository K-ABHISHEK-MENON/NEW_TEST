import numpy as np

def my_w_opt(g, phi, G, h):
    # g: (N,)
    # phi: (N,)
    # G: (N, m)
    # h: (m, 1)

    phi_diag = np.diag(phi)                 # (N, N)
    temp = g @ phi_diag @ G + h.flatten()   # ensure temp is (m,)
    w = temp.conj() / np.linalg.norm(temp)  # normalized beamforming vector
    return w
