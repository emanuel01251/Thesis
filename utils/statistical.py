import numpy as np
from scipy.stats import ttest_rel

# For SSP vs SOP without DA
ssp_metrics = {
    'precision': [44.00],  # Average precision
    'recall': [66.38],     # Average recall
    'f1': [50.05]          # Average F1
}

sop_metrics = {
    'precision': [46.15],  # Average precision
    'recall': [68.66],     # Average F1
    'f1': [52.27]          # Average F1
}

# Combine all metrics
ssp_combined = ssp_metrics['precision'] + ssp_metrics['recall'] + ssp_metrics['f1']
sop_combined = sop_metrics['precision'] + sop_metrics['recall'] + sop_metrics['f1']

# Calculate differences
differences = [sop - ssp for sop, ssp in zip(sop_combined, ssp_combined)]

# Calculate mean difference and standard deviation
mean_diff = np.mean(differences)
std_diff = np.std(differences, ddof=1)  # Using n-1 for degrees of freedom

# Calculate t-statistic with n=3 (3 metrics)
n = len(differences)  # = 3 for combined metrics
t_stat = (np.sqrt(n) * mean_diff) / std_diff
df = n - 1  # = 2 degrees of freedom

print(f"T-statistic: {t_stat:.4f}")
print(f"Degrees of freedom: {df}")