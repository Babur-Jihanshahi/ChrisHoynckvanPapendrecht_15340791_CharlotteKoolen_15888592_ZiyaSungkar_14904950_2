# performing a pairwise T-test

import pandas as pd
import numpy as np
from scipy.stats import t

# Read CSV file
data = pd.read_csv("question_2.csv")
sjf_data = pd.read_csv("mm1_fifo_sjf_comparison.csv")
n = 500

rhos = data["Rho"].values

sjf_mean = np.interp(rhos, sjf_data['Rho'], sjf_data['SJF_Mean'])
sjf_var = np.interp(rhos, sjf_data['Rho'], sjf_data['SJF_Var'])
means = np.column_stack((data[['Mean_1', 'Mean_2', 'Mean_4']].values, sjf_mean))
variances = np.column_stack((data[['Variance_1', 'Variance_2', 'Variance_4']].values, sjf_mean))

results = []

#do pairwise t-test
for i in range(means.shape[1]):
    for j in range(i + 1, means.shape[1]):

        mean_diff = means[:, i] - means[:, j]
        combined_variance = variances[:, i] / n + variances[:, j] / n
        
        # Compute the t-scores
        t_scores = mean_diff / np.sqrt(combined_variance)
        
        # Compute degrees of freedom
        df = (combined_variance**2) / (
            ((variances[:, i] / n)**2) / (n - 1) + ((variances[:, j] / n)**2) / (n - 1)
        )
        
        # Compute the p-values
        p_values = 2 * t.sf(np.abs(t_scores), df)

        group1 = f"Mean_{i+1}" if i < 3 else "SJF"
        group2 = f"Mean_{j+1}" if j < 3 else "SJF"
        
        results.append({
            "Group 1": group1,
            "Group 2": group2,
            "Rho_values": rhos.tolist(),
            "T-Scores": t_scores.tolist(),
            "P-Values": p_values.tolist()
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display or save results
print(results_df)
results_df.to_csv("t_test_results.csv", index=False)

filtered_results = []

# Iterate through each row
for _, row in results_df.iterrows():
    group_1 = row["Group 1"]
    group_2 = row["Group 2"]
    rho_values = row["Rho_values"]
    p_values = row["P-Values"]
    
    # Filter rho values where p-value > 0.05
    significant_rho = [rho for rho, p in zip(rho_values, p_values) if p > 0.0000000000000005]
    
    # If there are any significant rho values, store the result
    if significant_rho:
        filtered_results.append({
            "Group 1": group_1,
            "Group 2": group_2,
            "Significant Rho Values": significant_rho
        })

# Convert to DataFrame and display
filtered_results_df = pd.DataFrame(filtered_results)

# Display or save the results
print(filtered_results_df)
filtered_results_df.to_csv("significant_rho_values.csv", index=False)

# Print summary
print("\nStatistical Analysis Summary:")
print("============================")
for _, row in results_df.iterrows():
    print(f"\n{row['Group 1']} vs {row['Group 2']}:")
    significant_result = filtered_results_df[
        (filtered_results_df['Group 1'] == row['Group 1']) & 
        (filtered_results_df['Group 2'] == row['Group 2'])
    ]
    
    if not significant_result.empty:
        sig_rhos = significant_result.iloc[0]['Significant Rho Values']
        rho_ranges = f"ρ ∈ [{min(sig_rhos):.2f}, {max(sig_rhos):.2f}]"
        print(f"Configurations are NOT significantly different for {rho_ranges}")
    else:
        print("Configurations are significantly different at ALL rho values")
