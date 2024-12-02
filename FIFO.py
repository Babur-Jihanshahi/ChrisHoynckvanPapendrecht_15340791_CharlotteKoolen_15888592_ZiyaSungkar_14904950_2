import simpy 
import numpy as np
import csv
import visualize



#set true for M/D/1
DET=False
#set true for M/Lt/1
LT=True
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
        if DET: 
            tib = 1/time_in_bank 
        elif LT:
            if np.random.rand() < 0.75:
                tib = np.random.exponential(1 / (time_in_bank*2))
            else:
                tib = np.random.exponential(1 / (time_in_bank*0.4))
        else: 
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

    from SJF import run_simulation_sjf
    bins_experiments = []
    bin_countjes =[]
    average_wait =[]
    variance_wait = []

    
    if LT:
        rho = lambdaa/(mu) #for longtail the system load needs to be adjusted to average service time (twice original service time)
    else: 
        rho = lambdaa/mu #system load when working with one counter
    
    # If calculating distribuitons, also include for SJF (n=1)
    if calc_bins:
        all_waits_sjf = []
        trial_waits_sjf = []
        print("calculating distribution for sjf")
        for trial in range(num_trials):
            waitings_sjf = run_simulation_sjf(num_cust, lambdaa , mu, rand+trial)
            trial_waits_sjf.append(waitings_sjf)
        
        all_waits_sjf = np.concatenate(trial_waits_sjf)
        average_wait.append(np.mean(all_waits_sjf))
        max_wait = max(all_waits_sjf)  # Find the maximum value
        bins = np.arange(0, max_wait + bin_size, bin_size)
        counts, _ = np.histogram(all_waits_sjf, bins=bins)
        bin_counts = counts/num_trials
        bin_counts[bin_counts < 1e-2] = 0  #optional, if number of people in bin is very low, discard
        bins_experiments.append(bins)
        bin_countjes.append(bin_counts)

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
            bin_counts[bin_counts < 1e-2] = 0  #optional, if number of people in bin is very low, discard
            bins_experiments.append(bins)
            bin_countjes.append(bin_counts)

    return average_wait, variance_wait, bin_countjes, bins_experiments, rho

def iterate_rho(bin_size, mu, capacity, num_trials, num_cust, rand):
    '''
    Iterate over values of rhos, save each average result for that rho value by appending to a list and return
    ''' 
    # if LT:
    #     #for the longtail distribution the service rate on average is lower, thus the lambda needs to be lower for rho < 1
    #     lambdas = np.linspace(0.35, 0.4999, 100)
    # else:
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


def main(): 
    '''
    First choose the service distribution (set in global variables), choose longtail, exponential (default), or deterministic. 
    There are 3 different runs to choose from, one calculating the distribution of waiting times, one as examplatory run, and one
    iterating over different lambda (and thus rho values). all three these options perform the queueing for n=1, n=2 and n=4. 
    '''
    # setting variable values 
    rand = 43
    bin_size = 0.3
    num_cust = 500 # Total number of customers
    lambdaa = 0.99  # Generate new customers roughly every x seconds -> higher is quicker new arrivals
    mu = 1.0 #mu -> service time, lower is longer wait 
    capacity = [1, 2, 4]
    num_trials = 500


    # choose a calculation to perform, default calculation iterates over different rho values and calculates the average waiting times for n=1 n=2 and n=4 for the chosen distribution
    # calculate distribution of waiting times 
    calcdist = False

    # calculate an example run 
    example_run  = False


    # Choose out of Longtail, Deterministic and Exponential (default) service distribution
    if LT:
        dist = "$L_t$"
        print(r"Starting execution for longtail distribution (M/L_t/n)")
    elif DET:
        dist = "D"
        print(r"Starting execution for deterministic distribution (M/D/n)")
    else:
        dist = "M"
        print(r"Starting execution for exponential distribution (M/M/n)")

    if example_run:
            # do one examplatory run. 
        num_trials = 1
        waitings = []
        for i in range(3):
            waiting = run_simulation(num_cust, lambdaa*capacity[i], mu, capacity[i], rand)
            waitings.append(waiting)
        visualize.visualize_trial(waitings, distribution=dist)

    elif calcdist:
        # less to be calculated so increase number of trials to improve precision
        num_trials = 10000
        # calculate quantities that fall within the same bin of waiting time. shows distributions of waiting times. 
        average_wait, _, bin_countjes, bins_experiments, rho = initialize(bin_size, mu, lambdaa, capacity, num_trials, num_cust, rand, calc_bins=True)
        print(f"for capacity: 1 (SJF): the average wait time is: {average_wait[0]}")
        for k in range(len(capacity)):
            print(f"for capacity: {capacity[k]} (FIFO): the average wait time is: {average_wait[k+1]}")
        visualize.visualize_waiting(round(rho,2), capacity, mu,  lambdaa, bin_countjes, bin_size, bins_experiments, distribution= dist)

    else:
        # iterate over different values of rho, calculate the average waiting time in que over the number of customers, visualize the data. 
        means, variances, rhos = iterate_rho(bin_size, mu, capacity, num_trials, num_cust, rand)
        print(f"means: {len(means)}")
        print(f"variances: {len(variances)}")
        print(f"rhos: {len(rhos)}")

        with open(f"data/FIFO_{dist}.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
    
            # Write header
            writer.writerow(["Rho", "Mean_1", "Mean_2", "Mean_4", "Variance_1", "Variance_2", "Variance_4"])
    
            # Write data row by row
            for i, rho in enumerate(rhos):
                row = [rho] + [means[i][j] for j in range(3)] + [variances[i][j] for j in range(3)]
                writer.writerow(row)
        
        visualize.visualize_increasing_rho(np.column_stack(means), np.column_stack(variances), rhos, distribution=dist)


if __name__ == "__main__":
   main()
    