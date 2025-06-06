import os
import glob
import pandas as pd

# LLM name and base directory
llm_name = "mistral"
base_dir = f"results/BEA/cleaned/{llm_name}"
output_dir = f"results/BEA/consolidated/{llm_name}"
os.makedirs(output_dir, exist_ok=True)

# Process each iteration folder
iteration_dirs = sorted(glob.glob(os.path.join(base_dir, "iteration*")))

for iteration_dir in iteration_dirs:
    iteration_name = os.path.basename(iteration_dir)
    csv_files = glob.glob(os.path.join(iteration_dir, "*_output.csv"))

    combined_rows = []

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            combined_rows.append(df)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
        output_path = os.path.join(output_dir, f"{iteration_name}_combined.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
    else:
        print(f"No data found for {iteration_name}")
