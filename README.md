# ChrisHoynckvanPapendrecht_15340791_CharlotteKoolen_15888592_ZiyaSungkar_14904950_2
Second Assignment

Main Focus: Using Discrete Event Simulation (DES) to study queuing systems. 

1. **Theoretical Derivation**:
   - Look up or derive the theoretical result for the average waiting times in an M/M/n queue (multiple servers with FIFO scheduling) compared to an M/M/1 queue (single server).
   - Specifically, derive the result for \( n = 2 \).
   - Provide both a mathematical explanation of this result.

2. **Discrete Event Simulation (DES) Program**:
   - Write a DES program using SimPy to verify the theoretical waiting time results for \( n = 1, n = 2, \) and \( n = 4 \) servers.
   - Ensure that your simulation results have high statistical significance.
   - Investigate how the number of measurements needed to achieve statistical significance changes with the system load (\( \rho \)).

3. **Comparison with Shortest Job First Scheduling**:
   - Compare the results for the M/M/n queue (FIFO) with an M/M/1 queue where the scheduling is based on the shortest job first (i.e., prioritizing the smallest jobs).

4. **Experiment with Different Service Rate Distributions**:
   - Simulate systems with different service rate distributions:
     - **M/D/1 and M/D/n Queues**: where the service time is deterministic.
     - **Long-Tail Distribution**: where 75% of jobs follow an exponential distribution with an average service time of 1.0, and 25% have an average service time of 5.0 (this represents a hyperexponential distribution).
  
