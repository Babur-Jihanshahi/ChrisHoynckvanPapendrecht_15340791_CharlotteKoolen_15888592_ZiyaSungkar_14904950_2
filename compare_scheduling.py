import numpy as np
import matplotlib.pyplot as plt
import csv
from DES import run_simulation
from SJF import run_simulation_sjf

def compare_scheduling_policies(mu, lambdas, num_trials, num_cust, rand_seed):
    """
    Compare FIFO M/M/1 queue with SJF M/M/1 queue across different system loads
    
    Parameters:
        mu (float): Service rate
        lambdas (array): Array of arrival rates to test
        num_trials (int): Number of trials per configuration
        num_cust (int): Number of customers per trial
        rand_seed (int): Starting random seed
    """
    means_fifo = []
    means_sjf = []
    variances_fifo = []
    variances_sjf = []
    rhos = []

    for lam in lambdas:
        rho = lam/mu
        rhos.append(rho)
        print(f"\nProcessing rho {rho:.3f}")
        # FIFO M/M/1
        trial_waitings_fifo = []
        for trial in range(num_trials):
            waiting = run_simulation(num_cust, lam, mu, 1, rand_seed + trial)
            trial_waitings_fifo.append(waiting)
        all_waits_fifo = np.concatenate(trial_waitings_fifo)
        means_fifo.append(np.mean(all_waits_fifo))
        variances_fifo.append(np.var(all_waits_fifo))
        # SJF M/M/1
        trial_waitings_sjf = []
        for trial in range(num_trials):
            waiting = run_simulation_sjf(num_cust, lam, mu, rand_seed + trial)
            trial_waitings_sjf.append(waiting)
        all_waits_sjf = np.concatenate(trial_waitings_sjf)
        means_sjf.append(np.mean(all_waits_sjf))
        variances_sjf.append(np.var(all_waits_sjf))
    return np.array(means_fifo), np.array(variances_fifo), np.array(means_sjf), np.array(variances_sjf), np.array(rhos)

def visualize_comparison(means_fifo, variances_fifo, means_sjf, variances_sjf, rhos):
    """Visualize the comparison between FIFO and SJF scheduling"""
    plt.figure(figsize=(8, 6))
    # Plot FIFO results
    plt.plot(rhos, means_fifo, label='FIFO', marker='o', markersize=4)
    std_dev_fifo = np.sqrt(variances_fifo)
    plt.fill_between(rhos, 
                    np.maximum(0, means_fifo - std_dev_fifo),
                    means_fifo + std_dev_fifo,
                    alpha=0.2)
    # Plot SJF results
    plt.plot(rhos, means_sjf, label='SJF', marker='s', markersize=4)
    std_dev_sjf = np.sqrt(variances_sjf)
    plt.fill_between(rhos,
                    np.maximum(0, means_sjf - std_dev_sjf),
                    means_sjf + std_dev_sjf,
                    alpha=0.2)
    
    plt.xlabel('ρ')
    plt.ylabel('W')
    plt.title('Mean Waiting Time: FIFO vs SJF')
    plt.grid(True)
    plt.legend()
    plt.show()
    # Save results to CSV
    with open("scheduling_comparison.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Rho", "FIFO_Mean", "SJF_Mean", "FIFO_Var", "SJF_Var"]
        writer.writerow(header)
        for i, rho in enumerate(rhos):
            row = [rho, means_fifo[i], means_sjf[i], variances_fifo[i], variances_sjf[i]]
            writer.writerow(row)

if __name__ == "__main__":
    # Simulation parameters
    mu = 1.02
    lambdas = np.linspace(0.7, 0.95, 20)  # Testing up to ρ = 0.95
    num_trials = 500
    num_cust = 500
    rand_seed = 43
    
    means_fifo, variances_fifo, means_sjf, variances_sjf, rhos = compare_scheduling_policies(
        mu, lambdas, num_trials, num_cust, rand_seed
    )
    
    visualize_comparison(means_fifo, variances_fifo, means_sjf, variances_sjf, rhos)