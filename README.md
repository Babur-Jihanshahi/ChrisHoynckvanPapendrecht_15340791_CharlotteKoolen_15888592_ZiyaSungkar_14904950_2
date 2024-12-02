# ChrisHoynckvanPapendrecht_15340791_CharlotteKoolen_15888592_ZiyaSungkar_14904950_2
Second Assignment

Main Focus: Using Discrete Event Simulation (DES) to study queuing systems. Compare the performance between different distributions and number of servers. Also compares the performance of the FIFO and SJF service discipline. 

---

## Overview
`FIFO.py` is a Python script for simulating queueing systems with different service distributions and configurations. The script models the behavior of queues under First-In-First-Out (FIFO) discipline and allows analysis of waiting times for various scenarios.
`SJF.py` implements the shortest job first service discipline, and is only implemented for server. 

---

## Features
1. **Choose Service Distribution**:
   - Longtail (`M/L_t/n`)
   - Exponential (default: `M/M/n`)
   - Deterministic (`M/D/n`)

2. **Run Configurations**:
   - **Example Run**: Perform a single run to demonstrate queue behavior.
   - **Distribution Calculation**: Analyze the distribution of waiting times.
   - **Iterate Over Rho Values**: Simulate multiple runs for varying arrival rates (\(\lambda\)) to calculate average waiting times.

3. **Queue Capacities**:
   - Simulates queues for \(n = 1\), \(n = 2\), and \(n = 4\).

4. **Visualization and Data Export**:
   - Outputs graphs of waiting time distributions and rho analysis.
   - Saves results as a CSV file in the `data/` directory.

---

## How to Use
1. **Set Variables**:
   - Customize these variables directly in the script:
     - `rand`: Random seed for reproducibility.
     - `bin_size`: Bin size for calculating waiting time distributions.
     - `num_cust`: Total number of customers in the queue.
     - `lambdaa`: Arrival rate (\(\lambda\)).
     - `mu`: Service rate (\(\mu\)).
     - `capacity`: List of server capacities to simulate (\([1, 2, 4]\)). (only for FIFO.py)
     - `num_trials`: Number of simulation trials.

2. **Choose a Run Mode (Only for FIFO.py)**:
   - **Example Run**: Set `example_run = True` to perform a single illustrative run.
   - **Distribution Calculation**: Set `calcdist = True` to analyze the distribution of waiting times.
   - **Rho Iteration (Default)**: Leave both `example_run` and `calcdist` as `False` to iterate over different \(\rho = \lambda / \mu\) values.

3. **Select Service Distribution**:
   - Choose the distribution by setting one of the following global variables:
     - `LT = True` for Longtail.
     - `DET = True` for Deterministic.
     - Default is Exponential.

4. **Run the Script**:
   Execute the script in a Python environment:
   ```bash
   python FIFO.py
   python SJF.py
   python visualize.py
   python statistical_tests.py
