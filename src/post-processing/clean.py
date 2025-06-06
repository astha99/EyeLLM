import pandas as pd
import os
import glob
from src.main.similarityScores import (
    calculate_semantic_similarity, calculate_similarity,
    calculate_jaccard_similarity, calculate_f1_score
)
import openai
openai.api_key = "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"

# Define prefixes to strip
common_prefixes = [
    "Here's a possible continuation:",
    "Here is a possible continuation:",
    "Sure, here's a possible continuation:",
    "Continuing the text:",
    "Here's a potential continuation:",
"Here is a continuation of the essay:",
"Here's a continuation of your essay:",
"Here's the continuation:",
"Here's a continuation of the essay:",
"Here is a possible continuation of the essay:"

]

# Helper to strip leading phrases
def clean_completion(text, prefixes):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix):].strip()
    return text

# Original and new output base
old_base = "results/BEA/mistral"
new_base = "results/BEA/cleaned/mistral"
os.makedirs(new_base, exist_ok=True)

# Go through each iteration
for iteration in range(1, 11):
    print(f"Processing iteration {iteration}")
    old_dir = os.path.join(old_base, f"iteration_{iteration}")
    new_dir = os.path.join(new_base, f"iteration_{iteration}")
    os.makedirs(new_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(old_dir, "*_output.csv"))

    for file in csv_files:
        tokenid = os.path.basename(file).replace("_output.csv", "")
        out_file = os.path.join(new_dir, f"{tokenid}_output.csv")

        # Check if cleaned output already exists
        if os.path.exists(out_file):
            print(f"Skipping {tokenid}: cleaned file already exists.")
            continue

        df = pd.read_csv(file)

        # Clean completions
        for col in ["Pretext Only", "Pretext + Words", "Pretext + Sentences", "Control"]:
            df[f"{col}_Clean"] = df[col].apply(lambda x: clean_completion(x, common_prefixes))
            # print(df[["Pretext Only", "Pretext Only_Clean"]].head(5))

        # Recompute similarity scores
        for col in ["Pretext Only", "Pretext + Words", "Pretext + Sentences", "Control"]:
            base = f"{col}_Clean"
            df[f"{col} - Semantic"] = df.apply(lambda row: calculate_semantic_similarity(row["Student_Completion"], row[base]), axis=1)
            df[f"{col} - TFIDF"] = df.apply(lambda row: calculate_similarity(row["Student_Completion"], row[base]), axis=1)
            df[f"{col} - Jaccard"] = df.apply(lambda row: calculate_jaccard_similarity(row["Student_Completion"], row[base]), axis=1)
            df[f"{col} - F1"] = df.apply(lambda row: calculate_f1_score(row["Student_Completion"], row[base]), axis=1)

        # Save only the same columns as original (plus new scores)
        keep_cols = [
            "token_id", "Pretext0", "Pretextn", "Student_Completion"
            # "Pretext Only", "Pretext + Words", "Pretext + Sentences", "Control"
        ]
        cleaned_cols = [f"{col}_Clean" for col in ["Pretext Only", "Pretext + Words", "Pretext + Sentences", "Control"]]

        score_cols = [f"{col} - {metric}" for col in ["Pretext Only", "Pretext + Words", "Pretext + Sentences", "Control"]
                      for metric in ["Semantic", "TFIDF", "Jaccard", "F1"]]
        out_df = df[keep_cols + cleaned_cols + score_cols]

        # Save updated file
        out_df.to_csv(out_file, index=False)
        print(f"Saved cleaned and updated results to {out_file}")

