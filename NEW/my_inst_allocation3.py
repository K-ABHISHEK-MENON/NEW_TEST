
import numpy as np
from numpy.linalg import norm
from my_phi_opt import my_phi_opt
from my_w_opt import my_w_opt
from my_compute_sinr_k import my_compute_sinr_k

def my_inst_allocation3(m, K, N):
    np.random.seed(111)

    # Constants
    eta1 = 2.2
    eta2 = 2.2
    eta3 = 3.15
    d0 = 1
    c = 3e8
    f_c = 3.5e9
    lam = c / f_c
    d_s = lam / 2
    d_a_hat = lam / 2
    C_0 = c / (4 * np.pi * f_c * d0)

    BW = 100e6
    NF_dB = 10
    N_dBm = -174 + 10 * np.log10(BW) + NF_dB
    sig = 10 ** ((N_dBm - 30) / 10)

    ris_loc = np.array([48, 20, 0])
    bs_loc = np.array([0, 0, 0])
    user_locs = np.array([
        [45, 20, 0],
        [40, 20, 0],
        [35, 20, 0],
        [50, 20, 0],
        [55, 20, 0],
        [60, 20, 0],
    ])

    N_h = int(np.sqrt(N))
    N_v = int(np.sqrt(N))
    if N_h * N_v != N:
        raise ValueError("N must be a perfect square.")

    g_mat = np.zeros((N, K), dtype=complex)
    for k in range(K):
        user_pos = user_locs[k]
        d_vec = user_pos - ris_loc
        d_k_s = np.linalg.norm(d_vec)
        theta_k_s = np.arctan2(d_vec[1], d_vec[0])
        phi_k_s = np.arccos(d_vec[2] / d_k_s)
        tau_k_s = d_k_s / c
        f_k_s = 0

        idx_h = np.arange(N_h)
        idx_v = np.arange(N_v)
        a_h = np.exp(-1j * (2 * np.pi / lam) * d_s * idx_h * np.sin(theta_k_s) * np.cos(phi_k_s))
        a_v = np.exp(-1j * (2 * np.pi / lam) * d_s * idx_v * np.sin(theta_k_s) * np.cos(phi_k_s))
        a_R = np.kron(a_v[:, np.newaxis], a_h[np.newaxis, :]).flatten()

        g_L = np.exp(1j * 2 * np.pi * f_k_s * tau_k_s) * a_R
        g_NL = (1 / np.sqrt(2)) * (np.random.randn(N) + 1j * np.random.randn(N))
        g_mat[:, k] = np.sqrt(C_0 / (d_k_s ** eta1)) * (np.sqrt(10 / 11) * g_L + np.sqrt(1 / 11) * g_NL)

    kappa_bs = 10
    h_mat = np.zeros((m, K), dtype=complex)
    for k in range(K):
        user_pos = user_locs[k]
        d_vec = user_pos - bs_loc
        d_k_bs = np.linalg.norm(d_vec)
        theta_k_bs = np.arctan2(d_vec[1], d_vec[0])
        phi_k_bs = np.arccos(d_vec[2] / d_k_bs)
        a_B_bs = np.exp(-1j * (2 * np.pi / lam) * d_a_hat * np.arange(m)[:, None] * np.sin(theta_k_bs) * np.cos(phi_k_bs)).flatten()
        h_L = a_B_bs
        h_NL = (1 / np.sqrt(2)) * (np.random.randn(m) + 1j * np.random.randn(m))
        PL_k_bs = C_0 * d_k_bs ** (-eta3)
        h_mat[:, k] = np.sqrt(PL_k_bs) * (np.sqrt(kappa_bs / (kappa_bs + 1)) * h_L + np.sqrt(1 / (kappa_bs + 1)) * h_NL)

    rho_d = 15
    d_vec_G = bs_loc - ris_loc
    d_G = np.linalg.norm(d_vec_G)
    theta_bs = np.arctan2(d_vec_G[1], d_vec_G[0])
    phi_bs = np.arccos(d_vec_G[2] / d_G)
    PL_G = C_0 * d_G ** (-eta2)
    a_R = np.exp(-1j * (2 * np.pi / lam) * d_a_hat * np.arange(N) * np.sin(theta_bs) * np.cos(phi_bs))
    a_B = np.exp(-1j * (2 * np.pi / lam) * d_a_hat * np.arange(m) * np.sin(theta_bs) * np.cos(phi_bs))
    G_L = np.outer(a_R, a_B.conj())
    G_NL = (1 / np.sqrt(2)) * (np.random.randn(N, m) + 1j * np.random.randn(N, m))
    G = np.sqrt(PL_G) * (np.sqrt(rho_d / (rho_d + 1)) * G_L + np.sqrt(1 / (rho_d + 1)) * G_NL)

    dr_val_array = []
    max_iter = 10
    rate_user_matrix = np.zeros((K, max_iter))
    a_k_vecs = np.ones((N, K), dtype=int)
    w_mat = np.zeros((m, K), dtype=complex)

    for iter_index in range(max_iter):
        print(f"\n--- Iteration {iter_index + 1} ---")
        V_total = np.zeros((N + 1, N + 1), dtype=complex)
        w_mat_new = np.zeros((m, K), dtype=complex)

        for k in range(K):
            a_k = a_k_vecs[:, k]
        
            # 1. Compute phi_vec_init like MATLAB
            phi_vec_init = np.exp(1j * np.angle(g_mat[:, k] * (G @ h_mat[:, k])))

            # 2. Zero out entries not assigned to user k
            phi_vec_full = phi_vec_init.copy()
            assigned_idx = np.where(a_k != 0)[0]
            unassigned_idx = np.setdiff1d(np.arange(N), assigned_idx)
            phi_vec_full[unassigned_idx] = 0

            # 3. Compute beamformer using this phase
            w_mat_new[:, k] = my_w_opt(g_mat[:, k], phi_vec_full, G, h_mat[:, k])

            # 4. Optimize V_k using phi as initialization (optional if you want warm-starting)
            V_k = my_phi_opt(G, g_mat[:, k], h_mat[:, k], w_mat, sig, K, k, a_k)

        V_total += V_k

    V_total /= K


    for iter_index in range(max_iter):
        print(f"\n--- Iteration {iter_index + 1} ---")
        V_total = np.zeros((N + 1, N + 1), dtype=complex)
        w_mat_new = np.zeros((m, K), dtype=complex)

        for k in range(K):
            a_k = a_k_vecs[:, k]
            phi_vec_init = np.exp(1j * np.angle(g_mat[:, k] * (G @ h_mat[:, k])))
            
            V_k = my_phi_opt(G, g_mat[:, k], h_mat[:, k], w_mat, sig, K, k, a_k)
            phi_k = np.diag(V_k)[:-1]
            w_mat_new[:, k] = my_w_opt(g_mat[:, k], phi_k, G, h_mat[:, k])
            V_total += V_k

        V_total /= K

        total_rate = 0
        for k in range(K):
            a_k = a_k_vecs[:, k]
            sinr_k = my_compute_sinr_k(G, g_mat[:, k], h_mat[:, k], w_mat, V_total, sig, K, k, a_k)
            rate_k = np.log2(1 + sinr_k)
            rate_user_matrix[k, iter_index] = rate_k
            total_rate += rate_k
            print(f"User {k+1} SINR: {sinr_k:.4e}, Rate: {rate_k:.4f} bps/Hz")

        print(f"Sum Rate: {total_rate:.4f} bps/Hz")
        dr_val_array.append(total_rate)
        w_mat = w_mat_new.copy()

    return np.array(dr_val_array), rate_user_matrix[:, :max_iter], max_iter
