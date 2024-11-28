import simpy 
import numpy as np

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