
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

