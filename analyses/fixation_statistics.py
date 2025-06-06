import os
import glob
import re

import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
import pandas as pd
from dataWrangling import (
    get_completion,
    find_less_sustained_reading_examples,
    get_fixations,
    get_student_prompt
)

# Paths
json_files = glob.glob("data/json/*.json")
output_dir = "results/DataStats/"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "hesitation_events.csv")

# Collect all rows here
all_rows = []

# Loop through each JSON file
for json_file in json_files:
    tokenid = os.path.basename(json_file).replace("_incremental.json", "")
    student_prompt = get_student_prompt(tokenid, "data/promptid.csv")
    examples = find_less_sustained_reading_examples(json_file)
    fixated_data = get_fixations(json_file, examples, n=3)
    example_id = 1  # start from 1

    for entry in fixated_data:
        completion = get_completion(entry["pretext0"], entry["pretextn"])

        if not completion or not completion.strip():
            continue  # skip bad completions

        fixated_words = entry.get("fixated_words", [])
        fixated_sentences = entry.get("fixated_sentences", [])

        row = {
            "tokenid": tokenid,
            "example_id": example_id,
            "fixated_words": ", ".join(fixated_words),
            "fixated_sentences": " ".join(fixated_sentences),
            "non_fixated_words": ", ".join(entry.get("non_fixated_words", [])),
            "completion": completion,
            "num_fixated_words": len(fixated_words),
            "num_fixated_sentences": len(fixated_sentences),
            "completion_length": len(completion.split())  # word count
        }
        all_rows.append(row)
        example_id += 1

# Convert to DataFrame and save
df = pd.DataFrame(all_rows)
df.to_csv(output_file, index=False)
print(f"Combined file saved: {output_file}")

stats = {
    "fixated_words_avg": df["num_fixated_words"].mean(),
    "fixated_words_median": df["num_fixated_words"].median(),
    "fixated_sentences_avg": df["num_fixated_sentences"].mean(),
    "fixated_sentences_median": df["num_fixated_sentences"].median(),
    "completion_length_avg": df["completion_length"].mean(),
    "completion_length_median": df["completion_length"].median()
}

# Collect words from all completions
all_words = []
stop_words = set(stopwords.words("english"))
for text in df["completion"]:
    words = re.findall(r"\b\w+\b", text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    all_words.extend(filtered_words)

vocab_size = len(set(all_words))
stats["vocabulary_size"] = vocab_size

# Print summary
print("\n=== Descriptive Statistics for Dataset ===")
for key, value in stats.items():
    if key == "vocabulary_size":
        print(f"{key}: {value}")
    else:
        print(f"{key}: {value:.2f}")

