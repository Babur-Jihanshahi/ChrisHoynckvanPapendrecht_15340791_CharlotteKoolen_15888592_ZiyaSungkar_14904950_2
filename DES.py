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
import csv
import visualize

results = {} 

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
    #print('starting simulation')
    np.random.seed(random_seed)
    env = simpy.Environment()

    # Start processes and run
    waiting = []
    counter = simpy.Resource(env, capacity=num_servers)
    env.process(source(env, num_customers, arrival_rate, counter, service_rate, waiting))
    env.run()
    return waiting

def initialize(bin_size, mu, lambdaa, capacity, num_trials, num_cust, rand, calc_bins=False):
    '''
    Initialize, for a given lambda and mu, calculate average waiting time in queue for the specified number of trials
    If specified to do so, calculate the distribution of waiting times for a given bin size. 
    '''
    bins_experiments = []
    bin_countjes =[]
    average_wait =[]
    variance_wait = []

    rho = lambdaa/mu #system load when working with one counter
    for i in range(len(capacity)):
        lambda_using = lambdaa*capacity[i] #adjust mu such that the system load remains the same
        print(f'System Load: { rho}, and arrival rate: {lambda_using}')
        trial_waitings = []
        for _ in range(num_trials):
            waiting = run_simulation(num_cust, lambda_using , mu, capacity[i], rand)
            trial_waitings.append(waiting)
            #  update random variable
            rand+=1
        
        all_waits = np.concatenate(trial_waitings)
        max_wait = max(all_waits)  # Find the maximum value
        bins = np.arange(0, max_wait + bin_size, bin_size)

        # Initialize an array to accumulate bin counts
        bin_counts = np.zeros(len(bins) - 1)

        average_wait.append(np.mean(all_waits))
        variance_wait.append(np.var(all_waits))

        if calc_bins:
            counts, _ = np.histogram(all_waits, bins=bins)
            bin_counts = counts/num_trials
            bin_counts[bin_counts < 1e-3] = 0  #optional, if number of people in bin is very low, discard
            bins_experiments.append(bins)
            bin_countjes.append(bin_counts)

    return average_wait, variance_wait, bin_countjes, bins_experiments, rho

def iterate_rho(bin_size, mu, capacity, num_trials, num_cust, rand):
    '''
    Iterate over values of rhos, save each average result for that rho value by appending to a list and return
    ''' 
    lambdas = np.linspace(0.7, 0.9999, 100)
    all_average_waits = []
    all_variance_waits = []
    rhos = []
    for i in range(len(lambdas)):
        average_wait, var, _, _, rho = initialize(bin_size, mu, lambdas[i], capacity, num_trials, num_cust, rand)
        all_average_waits.append(average_wait)
        all_variance_waits.append(var)
        rhos.append(rho)
    return all_average_waits, all_variance_waits, rhos


if __name__ == "__main__":
    rand = 43
    bin_size = 0.3
    num_cust = 500 # Total number of customers
    lambdaa = 0.99  # Generate new customers roughly every x seconds -> higher is quicker new arrivals
    mu = 1.0 #mu -> service time, lower is longer wait 
    capacity = [1, 2, 4]
    num_trials = 500
    runone = True
    single_run = True

    if runone == True:
        if single_run == True:
            # do one examplatory run. 
            num_trials = 1
            waitings = []
            for i in range(3):
                waiting = run_simulation(num_cust, lambdaa*capacity[i], mu, capacity[i], rand)
                waitings.append(waiting)
            visualize.visualize_trial(waitings)

        else:
            # calculate quantities that fall within the same bin of waiting time. shows distributions of waiting times. 
            average_wait, var, bin_countjes, bins_experiments, rho = initialize(bin_size, mu, lambdaa, capacity, num_trials, num_cust, rand, calc_bins=True)
            for k in range(len(capacity)):
                print(f"for capacity: {capacity[k]}: the average wait time is: {average_wait[k]}")
            visualize.visualize_waiting(round(rho,2), capacity, mu,  lambdaa, bin_countjes, bin_size, bins_experiments)

    else:
        # iterate over different values of rho, calculate the average waiting time in que over the number of customers, visualize the data. 
        means, variances, rhos = iterate_rho(bin_size, mu, capacity, num_trials, num_cust, rand)
        print(f"means: {len(means)}")
        print(f"variances: {len(variances)}")
        print(f"rhos: {len(rhos)}")

        with open("question_2.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
    
            # Write header
            writer.writerow(["Rho", "Mean_1", "Mean_2", "Mean_4", "Variance_1", "Variance_2", "Variance_4"])
    
            # Write data row by row
            for i, rho in enumerate(rhos):
                row = [rho] + [means[i][j] for j in range(3)] + [variances[i][j] for j in range(3)]
                writer.writerow(row)
                
        visualize.visualize_increasing_rho(np.column_stack(means), np.column_stack(variances), rhos)
    