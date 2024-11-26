import matplotlib.pyplot as plt
import numpy as np

def visualize_waiting(rho, num_count, mu, lam, waiting_times, bin_size, bins):
    plt.figure(figsize=(4,2.5))
    for i in range(len(waiting_times)):
        plt.plot(bins[i][:-1], waiting_times[i], alpha = 1, label = f"N: {num_count[i]}")
    plt.title(rf"System Load: {rho}, $\mu$: {mu}, $\lambda$: {lam}")
    plt.xlabel("Waiting time")
    plt.ylabel("Number of People Waiting")
    plt.yscale("log")
    plt.legend()
    plt.show()


def visualize_increasing_rho(means, variances, rhos):
    # Plot each line with error bars
    plt.figure(figsize=(3, 3))
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
    plt.ylabel("W")
    plt.grid()
    plt.title("Mean and Variance for M/M/n")
    plt.show()

    
def visualize_trial(tot_waiting_time): 
    num_count = [1, 2, 4]
    plt.figure(figsize=(3,3))
    for i in range(3):
        people = np.linspace(0, len(tot_waiting_time[i]), len(tot_waiting_time[i]))
        plt.scatter(people, tot_waiting_time[i], s=1.5, label= f"n: {num_count[i]}", alpha=0.7)

    plt.title(r"Example $W_q$ for M/M/n")
    plt.legend()
    plt.xlabel("Customer")
    plt.ylabel(r"$W_q$")
    plt.xlim(0, len(tot_waiting_time[0]))
    plt.ylim(0)
    plt.show()


# if __name__ == "__main__":
#     visualize_waiting()
#     visualize_total_time()