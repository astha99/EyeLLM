import os
import glob
import pandas as pd
from dataWrangling import (
    get_completion, find_less_sustained_reading_examples, get_fixations, get_student_prompt
)
from src.main.GPTresponseGenerator import (
    generate_llm_responses, generate_llm_responses_with_fixated_words,
    generate_llm_responses_with_fixated_sentences, generate_llm_responses_control
)
from similarityScores import (
    calculate_semantic_similarity, calculate_similarity, calculate_jaccard_similarity, calculate_f1_score
)
import openai

openai.api_key = "######"

# Define input paths
json_files = glob.glob("data/json/*.json")
output_dir = "../../results/BEA/gpt-4/"
os.makedirs(output_dir, exist_ok=True)
# valid_examples = 0
# # saving the json for each token
# for json_file in json_files:
#     output_dir_json = "data/examples/"
#     os.makedirs(output_dir_json, exist_ok=True)
#
#     examples = find_less_sustained_reading_examples(json_file)
#     fixated_data = get_fixations(json_file, examples, n=3)
#     tokenid = os.path.basename(json_file).replace("_incremental.json", "")
#     student_prompt = get_student_prompt(tokenid, "data/promptid.csv")
#     output_file = os.path.join(output_dir_json, f"{tokenid}_data.json")
#     for entry in fixated_data:
#         student_completion = get_completion(entry['pretext0'], entry['pretextn'])
#         if not student_completion.strip():
#             continue
#         else:
#             valid_examples += 1
#     with open(output_file, "w") as f:
#         json.dump(fixated_data, f, indent=4)
#
# print(valid_examples)
# Process JSON files for multiple iterations
num_iterations = 10

for iteration in range(1, num_iterations + 1):
    print("Running iteration:", iteration)
    iteration_output_dir = os.path.join(output_dir, f"iteration_{iteration}")
    os.makedirs(iteration_output_dir, exist_ok=True)

    for json_file in json_files:
        tokenid = os.path.basename(json_file).replace("_incremental.json", "")
        output_file = os.path.join(iteration_output_dir, f"{tokenid}_output.csv")

        if os.path.exists(output_file):
            print(f"Skipping {json_file}, output already exists for iteration {iteration}.")
            continue
        print(f"Iteration {iteration} - Evaluating: {tokenid}")

        student_prompt = get_student_prompt(tokenid, "../../data/promptid.csv")
        examples = find_less_sustained_reading_examples(json_file)
        fixated_data = get_fixations(json_file, examples, n=3)

        data_rows = []
        valid_count = 0
        for entry in fixated_data:
            student_completion = get_completion(entry['pretext0'], entry['pretextn'])
            if not student_completion or not student_completion.strip():
                continue
            valid_count += 1

            responses = {
                "Pretext Only": generate_llm_responses(student_prompt, student_completion, entry['pretext0']),
                "Pretext + Words": generate_llm_responses_with_fixated_words(student_prompt, student_completion,
                                                                             entry['pretext0'], entry["fixated_words"]),
                "Pretext + Sentences": generate_llm_responses_with_fixated_sentences(student_prompt, student_completion,
                                                                                     entry['pretext0'],
                                                                                     entry["fixated_sentences"]),
                "Control": generate_llm_responses_control(student_prompt, student_completion, entry['pretext0'],
                                                          entry["non_fixated_words"])
            }

            scores = {case: [
                calculate_semantic_similarity(student_completion, responses[case]),
                calculate_similarity(student_completion, responses[case]),
                calculate_jaccard_similarity(student_completion, responses[case]),
                calculate_f1_score(student_completion, responses[case])
            ] for case in responses}

            data_rows.append([
                tokenid, entry['pretext0'], entry['pretextn'], student_completion,
                responses["Pretext Only"], responses["Pretext + Words"], responses["Pretext + Sentences"],
                responses["Control"],
                *[score for case in responses for score in scores[case]]
            ])

        columns = [
                      "token_id", "Pretext0", "Pretextn", "Student_Completion",
                      "Pretext Only", "Pretext + Words", "Pretext + Sentences", "Control"
                  ] + [f"{case} - {metric}" for case in responses for metric in ["Semantic", "TFIDF", "Jaccard", "F1"]]

        df = pd.DataFrame(data_rows, columns=columns)
        df.to_csv(output_file, index=False)
        print(f"Iteration {iteration} - Saved results to {output_file}")

# Compile results from all iterations
scores_all_iterations = []
data_all_iterations = []
id_info_file = "../../data/id_info.csv"
id_df = pd.read_csv(id_info_file)

for iteration in range(1, num_iterations + 1):
    iteration_output_dir = os.path.join(output_dir, f"iteration_{iteration}")
    compiled_output_file = f"{iteration_output_dir}/compiled_results.csv"
    output_file_full = f"{iteration_output_dir}/output_full.csv"
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
                df.drop(columns=["Pretext0", "Pretextn", "Student_Completion", "Pretext Only", "Pretext + Words",
                                 "Pretext + Sentences", "Control"], errors="ignore", inplace=True)
                scores_data.append(df)
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


# Save final compiled results from all iterations
final_output_full_all = os.path.join(output_dir, "output_full_all.csv")
final_compiled_output_all = os.path.join(output_dir, "compiled_results_all.csv")

if data_all_iterations:
    pd.concat(data_all_iterations, ignore_index=True).to_csv(final_output_full_all, index=False)
    print(f"Saved all full results to {final_output_full_all}")

if scores_all_iterations:
    compiled_results_all_df = pd.concat(scores_all_iterations, ignore_index=True)
    compiled_results_all_df.to_csv(final_compiled_output_all, index=False)
    print(f"Saved all compiled results to {final_compiled_output_all}")

    # Compute average scores while retaining identifier columns
    id_cols = ["participant_id", "token_id", "Topic", "example_id"]
    score_cols = [col for col in compiled_results_all_df.columns if col not in id_cols]

    df_avg = compiled_results_all_df.groupby(id_cols)[score_cols].mean().reset_index()

    # Save averaged results
    avg_output_file = os.path.join(output_dir, "averaged_results.csv")
    df_avg.to_csv(avg_output_file, index=False)
    print(f"Averaged results saved to {avg_output_file}")
