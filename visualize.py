import matplotlib.pyplot as plt
import numpy as np

def visualize_waiting(rho, num_count, mu, lam, waiting_times, bin_size, bins, distribution="M"):
    '''
    Visualizes quantities of customers falling inside the same waiting time bin. 
    plots a line for every n.
    '''

    plt.figure(figsize=(4,3.2), dpi=300)

    
    for i in range(len(num_count)):
        plt.plot(bins[i+1][:-1], waiting_times[i+1], alpha = 1, label = f"n: {num_count[i]}, FIFO")
    plt.plot(bins[0][:-1], waiting_times[0], alpha = 1, label = f"n: 1, SJF")
    
    plt.title(fr"Quantities of Queing Durations M/{distribution}/n ($\rho$: {rho})")
    plt.xlabel(r"$W_q$")
    plt.ylabel("Number of Customers")
    plt.yscale("log")
    plt.xlim(0, 150)
    plt.ylim(1e-2)
    plt.grid()
    plt.legend()
    # plt.savefig("queueing_quantities.png", dpi=300)
    plt.show()


def visualize_increasing_rho(means, variances, rhos, distribution="M"):
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
    plt.title(fr"Mean and Variance for M/{distribution}/n")
    # plt.savefig("mu1_num_cust500_numtrials500.png", dpi=300)
    plt.show()

def visualize_increasing_rho_sjf(means, variances, rhos):
    '''
    Visualizes for both sjf and fifo (for n =1) how the mean waiting time in que develops,
    while rho is increasing and approaching 1. Also plots the variance. 
    '''
    plt.figure(figsize=(3, 3), dpi=300)
    sd = ["FIFO", "SJF"]
    for i in range(len(means)):
        stddev = np.sqrt(variances[i])  # Convert variances to standard deviations
        mean = means[i]
        plt.plot(rhos, mean, label=f"service discipline: {sd[i]}")
    
        # Add the shaded area for variance
        lower_bound = np.array(mean) - stddev
        upper_bound = np.array(mean) + stddev
        plt.fill_between(rhos, lower_bound, upper_bound, alpha=0.2)

    plt.legend()
    plt.xlim(rhos[0], rhos[-1])
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$W_q$")
    plt.grid()
    plt.title(r"Mean and Variance for M/M/1/SJF vs M/M/1/FIFO")
    plt.ylim(0)
    # plt.savefig("mu1_num_cust500_numtrials500.png", dpi=300)
    plt.show()
    
def visualize_trial(tot_waiting_time, distribution="M", service=""): 
    '''
    Visualizes one trial, for every system (with different n).
    on the x-axes are the customers, and y-axes the waiting time in queue.
    '''
    num_count = [1, 2, 4]
    plt.figure(figsize=(3,3))
    for i in range(3):
        people = np.linspace(0, len(tot_waiting_time[i]), len(tot_waiting_time[i]))
        plt.scatter(people, tot_waiting_time[i], s=1.5, label=f"n: {num_count[i]}", alpha=0.7)

    plt.title(rf"Example $W_q$ for M/{distribution}/n{service}")
    plt.legend()
    plt.xlabel("Customer")
    plt.ylabel(r"$W_q$")
    plt.xlim(0, len(tot_waiting_time[0]))
    plt.ylim(0)
    # plt.savefig("example_trial.png", dpi=300)
    plt.show()

def visualize_function():
    '''
    Visualize the function of P_0, 
    '''
    rho = np.linspace(0.70, 0.9999, 1000)
    mu = 1.00
    Wq_mm1 = rho / (mu * (1 - rho))
    P0 = (1 + 2 * rho + ((2 * rho) ** 2) / (2 * (1 - rho))) ** (-1)
    Wq_mm2 = (rho**2 * P0) / (mu * (1 - rho)**2)

    plt.figure(figsize=(3, 3), dpi=300)
    plt.plot(rho, Wq_mm1, label=r'M/M/1', color='blue')
    plt.plot(rho, Wq_mm2, label=r'M/M/2', color='red', linestyle='--')
    plt.xlabel(r'System Load $\rho$')
    plt.ylabel(r'$W_q$')
    plt.title('Derived Waiting Time for M/M/1 and M/M/2')
    plt.xticks()
    plt.yticks()
    plt.xlim(0.7, 1)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 50)
    plt.show()

if __name__ == "__main__": 
    visualize_function()