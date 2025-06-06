import pandas as pd
import os

num_iterations = 10
output_dir = "../../results/BEA/cleaned/mistral"
id_info_file = "../../data/id_info.csv"
id_df = pd.read_csv(id_info_file)

scores_all_iterations = []
data_all_iterations = []

for iteration in range(1, num_iterations + 1):
    iteration_output_dir = os.path.join(output_dir, f"iteration_{iteration}")
    compiled_output_file = os.path.join(iteration_output_dir, "compiled_results_cleaned.csv")
    output_file_full = os.path.join(iteration_output_dir, "output_full_cleaned.csv")

    # Skip this iteration if files already exist
    if os.path.exists(compiled_output_file) and os.path.exists(output_file_full):
        print(f"Iteration {iteration} - Skipping, files already exist.")
        scores_all_iterations.append(pd.read_csv(compiled_output_file))
        data_all_iterations.append(pd.read_csv(output_file_full))
        continue

    scores_data = []
    all_data = []

    for _, row in id_df.iterrows():
        participant_id = row["Participant"]
        for col in ["A", "B"]:
            token_id = row[col]
            input_file = os.path.join(iteration_output_dir, f"{token_id}_output.csv")
            if os.path.exists(input_file):
                df = pd.read_csv(input_file)
                df.insert(0, "participant_id", participant_id)
                df.insert(2, "Topic", col)
                df.insert(3, "example_id", range(1, len(df) + 1))
                all_data.append(df)

                # Create a trimmed copy for scores
                df_scores = df.drop(columns=[
                    "Pretext0", "Pretextn", "Student_Completion",
                    "Pretext Only_Clean", "Pretext + Words_Clean", "Pretext + Sentences_Clean", "Control_Clean"
                ], errors="ignore").copy()

                scores_data.append(df_scores)

            else:
                print(f"Iteration {iteration} - Warning: {input_file} not found.")

    if all_data:
        final_df_full = pd.concat(all_data, ignore_index=True)
        final_df_full.to_csv(output_file_full, index=False)
        print(f"Iteration {iteration} - Saved full results to {output_file_full}")
        data_all_iterations.append(final_df_full)

    if scores_data:
        final_df = pd.concat(scores_data, ignore_index=True)
        final_df.to_csv(compiled_output_file, index=False)
        print(f"Iteration {iteration} - Compiled results saved to {compiled_output_file}")
        scores_all_iterations.append(final_df)

# Final consolidated files
final_output_full_all = os.path.join(output_dir, "output_full_all_cleaned.csv")
final_compiled_output_all = os.path.join(output_dir, "compiled_results_all_cleaned.csv")

if data_all_iterations:
    pd.concat(data_all_iterations, ignore_index=True).to_csv(final_output_full_all, index=False)
    print(f"Saved all full cleaned results to {final_output_full_all}")

if scores_all_iterations:
    compiled_results_all_df = pd.concat(scores_all_iterations, ignore_index=True)
    compiled_results_all_df.to_csv(final_compiled_output_all, index=False)
    print(f"Saved all compiled cleaned results to {final_compiled_output_all}")

    # Averaging
    id_cols = ["participant_id", "token_id", "Topic", "example_id"]
    # Only use numeric columns for averaging (score columns)
    numeric_cols = compiled_results_all_df.select_dtypes(include='number').columns.tolist()
    score_cols = [col for col in numeric_cols if col not in id_cols]

    df_avg = compiled_results_all_df.groupby(id_cols)[score_cols].mean().reset_index()

    avg_output_file = os.path.join(output_dir, "averaged_results_cleaned.csv")
    df_avg.to_csv(avg_output_file, index=False)
    print(f"Averaged cleaned results saved to {avg_output_file}")
