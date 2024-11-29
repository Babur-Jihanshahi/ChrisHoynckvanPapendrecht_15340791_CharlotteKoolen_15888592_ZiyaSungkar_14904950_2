# performing a pairwise T-test

import pandas as pd
import numpy as np
from scipy.stats import t


# Read CSV files
data_MMn = pd.read_csv("question_2.csv")
data_MDn = pd.read_csv("question_4_Det.csv")
data_MLtn = pd.read_csv("question_4_longtail.csv")

rhos_MMn = data_MMn["Rho"].values
means_MMn = data_MMn[["Mean_1", "Mean_2", "Mean_4"]].values
variances_MMn = data_MMn[["Variance_1", "Variance_2", "Variance_4"]].values
results = []

rhos_MDn = data_MDn["Rho"].values
means_MDn = data_MDn[["Mean_1", "Mean_2", "Mean_4"]].values
variances_MDn = data_MDn[["Variance_1", "Variance_2", "Variance_4"]].values

rhos_MLtn = data_MLtn["Rho"].values
means_MLtn = data_MLtn[["Mean_1", "Mean_2", "Mean_4"]].values
variances_MLtn = data_MLtn[["Variance_1", "Variance_2", "Variance_4"]].values

n = 500
results = []

# Pairwise t-tests
for i in range(means_MMn.shape[1]):
    pairs = [
        (means_MMn[:, i], variances_MMn[:, i], "M/M/n"),
        (means_MDn[:, i], variances_MDn[:, i], "M/D/n"),
        (means_MLtn[:, i], variances_MLtn[:, i], "M/Lt/n")
    ]

    # Pairwise comparisons between groups
    for p1 in range(len(pairs)):
        for p2 in range(p1 + 1, len(pairs)):
            mean_diff = pairs[p1][0] - pairs[p2][0]
            combined_variance = pairs[p1][1] / n + pairs[p2][1] / n
        
        # Compute the t-scores
        t_scores = mean_diff / np.sqrt(combined_variance)
        
        # Compute degrees of freedom
        df = (combined_variance**2) / (
                ((pairs[p1][1] / n)**2) / (n - 1) +
                ((pairs[p2][1] / n)**2) / (n - 1)
            )
        
        # Compute the p-values
        p_values = 2 * t.sf(np.abs(t_scores), df)
        
        results.append({
                "Group 1": f"{pairs[p1][2]} Mean_{i+1}",
                "Group 2": f"{pairs[p2][2]} Mean_{i+1}",
                "Rho_values": rhos_MMn.tolist(),
                "T-Scores": t_scores.tolist(),
                "P-Values": p_values.tolist()
            })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display or save results
print(results_df)
results_df.to_csv("t_test_results_MMn_MDn_MLtn.csv", index=False)

filtered_results = []

# Iterate through each row
for _, row in results_df.iterrows():
    group_1 = row["Group 1"]
    group_2 = row["Group 2"]
    rho_values = row["Rho_values"]
    p_values = row["P-Values"]
    
    # Filter rho values where p-value < 0.05
    significant_rho = [rho for rho, p in zip(rho_values, p_values) if p < 0.05]
    
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
filtered_results_df.to_csv("significant_rho_values_MMn_MDn_MLtn.csv", index=False)


