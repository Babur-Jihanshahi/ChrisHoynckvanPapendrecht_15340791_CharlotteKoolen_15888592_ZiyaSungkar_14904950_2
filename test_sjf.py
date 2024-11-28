import numpy as np
from SJF import run_simulation_sjf
from DES import run_simulation  # For FIFO simulation

def test_sjf_vs_fifo():
    """Compare SJF directly against FIFO simulation"""
    num_customers = 5000
    mu = 1.0
    rho = 0.5
    seed = 42
    
    # Run both simulations
    waiting_times_sjf = run_simulation_sjf(num_customers, rho, mu, seed)
    waiting_times_fifo = run_simulation(num_customers, rho, mu, 1, seed)
    
    mean_wait_sjf = np.mean(waiting_times_sjf)
    mean_wait_fifo = np.mean(waiting_times_fifo)
    
    print(f"\nComparison Debug:")
    print(f"SJF mean wait: {mean_wait_sjf:.4f}")
    print(f"FIFO mean wait: {mean_wait_fifo:.4f}")
    print(f"Improvement: {((mean_wait_fifo - mean_wait_sjf)/mean_wait_fifo)*100:.2f}%")
    assert mean_wait_sjf < mean_wait_fifo, "SJF should have lower mean waiting time than FIFO"

if __name__ == "__main__":
    test_sjf_vs_fifo()