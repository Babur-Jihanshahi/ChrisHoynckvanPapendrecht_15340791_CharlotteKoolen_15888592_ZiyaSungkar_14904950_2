import matplotlib.pyplot as plt
import numpy as np

def visualize_waiting(rho, num_count, mu, lam, waiting_times, bin_size, bins):
    '''
    Visualizes quantities of customers falling inside the same waiting time bin. 
    plots a line for every n.
    '''

    plt.figure(figsize=(4,3.2), dpi=300)
    for i in range(len(waiting_times)):
        plt.plot(bins[i][:-1], waiting_times[i], alpha = 1, label = f"n: {num_count[i]}")
    plt.title(fr"Quantities of Queing Durations ($\rho$: {rho})")
    plt.xlabel(r"$W_q$")
    plt.ylabel("Number of Customers")
    plt.yscale("log")
    plt.xlim(0)
    plt.legend()
    # plt.savefig("queueing_quantities.png", dpi=300)
    plt.show()


def visualize_increasing_rho(means, variances, rhos):
    '''
    Visualizes for every system (with different n) how the mean waiting time in que develops,
    while rho is increasing and approaching 1. Also plots the variance. 
    '''
    plt.figure(figsize=(3, 3), dpi=300)
    values= [1, 2, 4]
    for i in range(len(means)):
        stddev = np.sqrt(variances[i])  # Convert variances to standard deviations
        mean = means[i]
        plt.plot(rhos, mean, label=f"n: {values[i]}")
    
        # Add the shaded area for variance
        lower_bound = np.array(mean) - stddev
        upper_bound = np.array(mean) + stddev
        plt.fill_between(rhos, lower_bound, upper_bound, alpha=0.2)

    plt.legend()
    plt.xlim(rhos[0], rhos[-1])
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$W_q$")
    plt.grid()
    plt.title(r"Mean and Variance for M/$L_t$/n")
    # plt.savefig("mu1_num_cust500_numtrials500.png", dpi=300)
    plt.show()
    
def visualize_trial(tot_waiting_time): 
    '''
    Visualizes one trial, for every system (with different n).
    on the x-axes are the customers, and y-axes the waiting time in queue.
    '''
    num_count = [1, 2, 4]
    plt.figure(figsize=(3,3))
    for i in range(3):
        people = np.linspace(0, len(tot_waiting_time[i]), len(tot_waiting_time[i]))
        plt.scatter(people, tot_waiting_time[i], s=1.5, label=f"n: {num_count[i]}", alpha=0.7)

    plt.title(r"Example $W_q$ for M/M/n and SJF")
    plt.legend()
    plt.xlabel("Customer")
    plt.ylabel(r"$W_q$")
    plt.xlim(0, len(tot_waiting_time[0]))
    plt.ylim(0)
    # plt.savefig("example_trial.png", dpi=300)
    plt.show()
