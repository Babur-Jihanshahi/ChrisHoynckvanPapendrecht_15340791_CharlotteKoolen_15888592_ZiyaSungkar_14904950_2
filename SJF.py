import simpy
import numpy as np

class SJFQueue(simpy.PriorityResource):
    """Custom resource implementing Shortest Job First scheduling"""
    def __init__(self, env, capacity=1):
        super().__init__(env, capacity)

    def request(self, service_time):
        """Override request to use service time as priority"""
        return super().request(priority=service_time)

def source_sjf(env, number, interval, counter, time_in_bank, waiting):
    """Source generates customers randomly for SJF queue"""
    for i in range(number):
        service_time = np.random.exponential(1.0 / time_in_bank)
        c = customer_sjf(env, f"Customer {i:02d}", counter, service_time, waiting)
        env.process(c)
        t = np.random.exponential(1.0 / interval)
        yield env.timeout(t)

def customer_sjf(env, name, counter, service_time, waiting):
    """Customer arrives, is served and leaves under SJF scheduling"""
    