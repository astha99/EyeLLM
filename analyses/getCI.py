import pandas as pd
import scipy.stats as stats

# Load your long-format CSV
df = pd.read_csv("results/BEA/compiled/cleaned/mistral.csv")  # Update path if needed

# Function to compute mean, std, and 95% CI
def compute_ci(group):
    mean = group["similarity_score"].mean()
    std = group["similarity_score"].std()
    n = group["similarity_score"].count()
    stderr = std / (n ** 0.5)
    ci_margin = stats.t.ppf(0.975, df=n-1) * stderr if n > 1 else 0
    return pd.Series({
        "mean": mean,
        "std": std,
        "count": n,
        "stderr": stderr,
        "95% CI Lower": mean - ci_margin,
        "95% CI Upper": mean + ci_margin
    })

# Group and compute statistics
ci_df = df.groupby(["condition", "measure"], group_keys=False).apply(compute_ci).reset_index()

# Save to CSV
ci_df.to_csv("results/BEA/ci/mistral_ci_stats.csv", index=False)
