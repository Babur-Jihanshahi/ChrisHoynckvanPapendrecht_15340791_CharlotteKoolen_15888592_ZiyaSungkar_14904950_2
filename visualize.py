import matplotlib.pyplot as plt
import numpy as np

def visualize_waiting(rho, num_count, mu, lam, waiting_times, bin_size, bins):
    plt.figure(figsize=(4,2.5))
    for i in range(len(waiting_times)):
        plt.plot(bins[i][:-1], waiting_times[i], alpha = 1, label = f"N: {num_count[i]}")
    plt.title(rf"System Load: {rho}, $\mu$: {mu}, $\lambda$: {lam}")
    plt.xlabel("Waiting time")
    plt.ylabel("Number of People Waiting this long")
    plt.yscale("log")
    plt.legend()
    plt.show()

    
    
def visualize_total_time(results, num_count): 
    for i in range(len(results)):
        tot_waiting_time = [res["waiting_time"] + res["service_time"] for res in results[i].values()]
        people = np.linspace(0, len(tot_waiting_time), len(tot_waiting_time))
        plt.scatter(people, tot_waiting_time, s=0.5, label= f"N: {num_count[i]}", alpha=0.7)
    pass


# if __name__ == "__main__":
#     visualize_waiting()
#     visualize_total_time()