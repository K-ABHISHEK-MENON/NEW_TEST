
import numpy as np
import matplotlib.pyplot as plt
from my_inst_allocation3 import my_inst_allocation3

def main():
    m = 2
    K = 2
    N = 16

    print("Beginning instantaneous allocation")
    dr_val_array, rate_user_matrix, iter_index = my_inst_allocation3(m, K, N)

    print("Finished allocation.")
    print("Final sum rates per iteration:", dr_val_array)
    print("Rate matrix shape:", rate_user_matrix.shape)

    # Plot Sum Data Rate
    plt.figure()
    plt.plot(range(1, len(dr_val_array) + 1), dr_val_array, '-ok', linewidth=2)
    plt.xlabel('Iteration Index')
    plt.ylabel('Sum Data Rate (bps/Hz)')
    plt.title('Total Sum Rate over Iterations')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sum_data_rate.png')
    plt.show()

    # Plot Per-User Data Rates
    plt.figure()
    for k in range(K):
        plt.plot(range(1, iter_index + 1), rate_user_matrix[k, :iter_index], linewidth=2, label=f'User {k+1}')

    avg_rate = np.mean(rate_user_matrix[:, :iter_index], axis=0)
    plt.plot(range(1, iter_index + 1), avg_rate, 'k--', linewidth=2.5, label='Average')

    plt.xlabel('Iteration Index')
    plt.ylabel('Data Rate (bps/Hz)')
    plt.title('Per-User and Average Data Rates over Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('per_user_data_rate.png')
    plt.show()

if __name__ == "__main__":
    main()





import numpy as np
import cvxpy as cp

def my_phi_opt(G, g, h, w_mat, sig, num_users, curr_user_idx, a):
    h = h.reshape(-1, 1)
    epsilon = 1e-6
    lam = 0
    a_idx = np.where(a == 1)[0]
    V_dim = len(a_idx) + 1

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

    den = 0
    for i in range(num_users):
        if i != curr_user_idx:
            w_i = w_mat[:, i].reshape(-1, 1)
            Q_tilde_i = construct_Q_tilde(w_i)
            den += np.trace(Q_tilde_i).real
    den += sig
    lam = np.trace(Q_tilde).real / den

    F = np.inf
    while F > epsilon:
        V_real = cp.Variable((V_dim, V_dim))
        V_imag = cp.Variable((V_dim, V_dim))
        V = V_real + 1j * V_imag
        num_expr = cp.real(cp.trace(Q_tilde @ V))

        den_expr = 0
        for i in range(num_users):
            if i != curr_user_idx:
                w_i = w_mat[:, i].reshape(-1, 1)
                Q_tilde_i = construct_Q_tilde(w_i)
                den_expr += cp.real(cp.trace(Q_tilde_i @ V))
        den_expr += sig

        objective = cp.Maximize(num_expr - lam * den_expr)
        constraints = [V >> 0, cp.diag(V) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        num_val = np.trace(Q_tilde @ V.value).real
        den_val = 0
        for i in range(num_users):
            if i != curr_user_idx:
                w_i = w_mat[:, i].reshape(-1, 1)
                Q_tilde_i = construct_Q_tilde(w_i)
                den_val += np.trace(Q_tilde_i @ V.value).real
        den_val += sig
        F = num_val - lam * den_val
        lam = num_val / den_val

    return V.value

