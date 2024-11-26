"""
Bank renege example

Covers:

- Resources: Resource
- Condition events

Scenario:
  A counter with a random service time and customers who renege. Based on the
  program bank08.py from TheBank tutorial of SimPy 2. (KGM)

"""


import simpy 
import matplotlib.pyplot as plt
import numpy as np
import visualize

results = {} 
WAITING = []

def source(env, number, interval, counter, time_in_bank, waiting):
    """Source generates customers randomly"""
    for i in range(number):
        c = customer(env, f'Customer {i:02d}', counter, time_in_bank, waiting)
        env.process(c)
        t = np.random.exponential(1.0 / interval)
        yield env.timeout(t)


def customer(env, name, counter, time_in_bank, waiting):
    """Customer arrives, is served and leaves."""
    arrive = env.now
    # print(f'{arrive:7.4f} {name}: Here I am')

    with counter.request() as req:
        # Wait for the counter 
        yield req
        wait = env.now - arrive

        # print(f'{env.now:7.4f} {name}: Waited {wait:6.3f}')

        tib = np.random.exponential(1.0 / time_in_bank)
        yield env.timeout(tib)
        # print(f'{env.now:7.4f} {name}: Finished')
        results[name] = {
            "arrival_time": arrive,
            "waiting_time": wait,
            "service_time": tib,
            "departure_time": env.now
        }
        waiting.append(wait)



def run_simulation(num_customers, arrival_rate, service_rate, num_servers, random_seed=42):
    """
    Run the discrete-event simulation with configurable parameters.

    Parameters:
        num_customers (int): Total number of customers.
        arrival_rate (float): Rate of arrivals (lambda).
        service_rate (float): Rate of service (mu).
        num_servers (int): Number of servers (n).
        random_seed (int): Random seed for reproducibility.

    Returns:
        dict: Results dictionary with metrics for each customer.
    """
    # Setup and start the simulation
    print('starting simulation')
    np.random.seed(random_seed)
    env = simpy.Environment()

    # Start processes and run
    waiting = []
    counter = simpy.Resource(env, capacity=num_servers)
    env.process(source(env, num_customers, arrival_rate, counter, service_rate, waiting))
    env.run()
    return waiting

def initialize(bin_size, mu, lambdaa, capacity, num_trials, num_cust, rand):
    bins_experiments = []
    bin_countjes =[]
    average_wait =[]
    for i in range(len(capacity)):
        rho = lambdaa/(mu) #system load when working with one counter
        mu_using = lambdaa/(rho*capacity[i]) #adjust mu such that the system load remains the same
        print(f'System Load: { rho}, and rate to serve: {mu_using}')
        trial_waitings = []
        for _ in range(num_trials):
            waiting = run_simulation(num_cust, lambdaa, mu_using, capacity[i], rand)
            trial_waitings.append(waiting)
            #  update random variable
            rand+=1
        
        all_waits = np.concatenate(trial_waitings)
        max_wait = max(all_waits)  # Find the maximum value
        bins = np.arange(0, max_wait + bin_size, bin_size)

        # Initialize an array to accumulate bin counts
        bin_counts = np.zeros(len(bins) - 1)
        average_wait.append(np.mean(all_waits))
        counts, _ = np.histogram(all_waits, bins=bins)
        bin_counts = counts/num_trials
        bins_experiments.append(bins)
        bin_countjes.append(bin_counts)

    return average_wait, bin_countjes, bins_experiments, rho

if __name__ == "__main__":
    rand = 43
    bin_size = 0.1
    num_cust = 10 # Total number of customers
    lambdaa = 1.0  # Generate new customers roughly every x seconds -> lower is quicker new arrivals
    mu = 1.002 #mu -> lower is longer wait 
    capacity = [1, 2, 4]
    num_trials = 10000    
    average_wait, bin_countjes, bins_experiments, rho = initialize(bin_size, mu, lambdaa, capacity, num_trials, num_cust, rand)
    
    
    for k in range(len(capacity)):
        print(f"for capacity: {capacity[k]}: the average wait time is: {average_wait[k]}")

    visualize.visualize_waiting(round(rho,2), capacity, mu,  lambdaa, bin_countjes, bin_size, bins_experiments)