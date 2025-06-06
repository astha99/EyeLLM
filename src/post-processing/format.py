import pandas as pd

# Load your data
df = pd.read_csv("../../results/BEA/gpt-4/compiled_results_all.csv")  # Replace with your actual file

# Melt the dataframe
df_long = df.melt(
    id_vars=["participant_id", "token_id", "Topic", "example_id"], # Add Topic
    var_name="condition_measure",
    value_name="similarity_score"
)

# Extract condition and measure
df_long[["condition", "measure"]] = df_long["condition_measure"].str.rsplit(" - ", n=1, expand=True)

# Remove TFIDF rows
df_long = df_long[df_long["measure"] != "TFIDF"]

# Rename columns
df_long = df_long[["participant_id", "token_id", "Topic", "example_id", "condition", "measure", "similarity_score"]]

# Save or display
df_long.to_csv("results/BEA/compiled/gpt4.csv", index=False)  # Save to a file if needed
print(df_long.head())  # Display first few rows
