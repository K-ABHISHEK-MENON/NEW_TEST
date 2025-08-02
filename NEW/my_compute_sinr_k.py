
import numpy as np

def my_compute_sinr_k(G, g, h, w_mat, V, sig, num_users, curr_user_idx, a):
    h = h.reshape(-1, 1)
    a_idx = np.where(a == 1)[0]

    def construct_Q_tilde(w):
        Q_full = (np.diag(g) @ G @ w) @ ((w.conj().T @ G.conj().T @ np.diag(g).conj()))
        c_full = (np.diag(g).conj().T @ G @ w @ w.conj().T @ h.conj()).flatten()
        inner = (h.conj().T @ w)[0, 0]
        d_val = np.real(inner * inner.conj())

        Q = Q_full[np.ix_(a_idx, a_idx)]
        c = c_full[a_idx]
        Q_tilde = np.block([
            [Q, c[:, np.newaxis]],
            [c.conj()[np.newaxis, :], np.array([[d_val]])]
        ])
        return Q_tilde

    w = w_mat[:, curr_user_idx].reshape(-1, 1)
    Q_tilde = construct_Q_tilde(w)
    num = np.trace(Q_tilde @ V).real

    den = 0
    for i in range(num_users):
        if i != curr_user_idx:
            w_i = w_mat[:, i].reshape(-1, 1)
            Q_tilde_i = construct_Q_tilde(w_i)
            den += np.trace(Q_tilde_i @ V).real
    den += sig

    
