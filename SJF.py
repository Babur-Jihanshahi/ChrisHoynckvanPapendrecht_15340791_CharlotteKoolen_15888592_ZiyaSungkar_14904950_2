import simpy 
import numpy as np
import csv
from FIFO import run_simulation
import visualize

class SJFQueue(simpy.PriorityResource):
    """Custom resource implementing Shortest Job First scheduling"""
    def __init__(self, env, capacity=1):
        super().__init__(env, capacity)

    def request(self, service_time):
        """Override request to use service time as priority
        Note: Lower priority values are served first in SimPy"""
        return super().request(priority=service_time)

def source_sjf(env, number, interval, counter, time_in_bank, waiting):
    """Source generates customers randomly for SJF queue"""
    for i in range(number):
        # Generate service time upfront for SJF scheduling
        service_time = np.random.exponential(1.0 / time_in_bank)
        c = customer_sjf(env, f"Customer {i:02d}", counter, service_time, waiting)
        env.process(c)
        # Time until next arrival
        t = np.random.exponential(1.0 / interval)
        yield env.timeout(t)

def customer_sjf(env, name, counter, service_time, waiting):
    """Customer arrives, is served and leaves under SJF scheduling"""
    arrive = env.now
    # Request service with priority based on service time
    with counter.request(service_time) as req:
        # Wait for service
        yield req
        wait = env.now - arrive
        # Service time
        yield env.timeout(service_time)
        # Store both waiting time and service time for analysis
        waiting.append((wait, service_time))

def run_simulation_sjf(num_customers, arrival_rate, service_rate, random_seed=42):
    """Run SJF simulation with debugging info"""
    np.random.seed(random_seed)
    env = simpy.Environment()
    
    waiting = []
    counter = SJFQueue(env)
    env.process(source_sjf(env, num_customers, arrival_rate, counter, service_rate, waiting))
    env.run()
    # Separate waiting times and service times
    waits, services = zip(*waiting) if waiting else ([], [])
    return list(waits)

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
    

if __name__ == "__main__":
    # Simulation parameters
    mu = 1.00
    lambdas = np.linspace(0.7, 0.9999, 100)  # Testing up to ρ = 0.95
    num_trials = 500
    num_cust = 500
    rand_seed = 43
    
    means_fifo, variances_fifo, means_sjf, variances_sjf, rhos = compare_scheduling_policies(
        mu, lambdas, num_trials, num_cust, rand_seed
    )
    

    visualize.visualize_increasing_rho_sjf([means_fifo, means_sjf], [variances_fifo, variances_sjf], rhos)
    # visualize_comparison(means_fifo, variances_fifo, means_sjf, variances_sjf, rhos)
    with open("data/SJF_M.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Rho", "FIFO_Mean", "SJF_Mean", "FIFO_Var", "SJF_Var"]
        writer.writerow(header)
        for i, rho in enumerate(rhos):
            row = [rho, means_fifo[i], means_sjf[i], variances_fifo[i], variances_sjf[i]]
            writer.writerow(row)