import json
import re
import string

from nltk.corpus import stopwords
import pandas as pd
import random

# Get the list of English stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"<-->|<-->", "", text)  # Remove fixation markers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    return text.strip().lower()

def get_student_prompt(tokenid, csv_path):
    """
    Extracts the student prompt based on token ID.
    If token ID is in column A of CSV, return prompt1;
    if token ID is in column B, return prompt2.
    """
    # Define the prompts
    prompt1 = "Some people have said that finding and implementing green technologies, such as wind or solar power, should be the focus of our efforts to avert climate crisis. To what extent do you agree or disagree with this statement? Try to support your arguments with appropriate evidence from, for example, your knowledge of scientific evidence, your own experience, or your observations and reading"
    prompt2 = "Some people have argued that animals should be given similar rights to humans. To what extent do you agree or disagree with this statement? Try to support your arguments with appropriate evidence from, for example, your knowledge of scientific evidence, your own experience, or your observations and reading"

    # Load CSV file
    df = pd.read_csv(csv_path)

    # Check if tokenid is in column A or B
    if tokenid in df.iloc[:, 0].values:  # Check first column (A)
        return prompt1
    elif tokenid in df.iloc[:, 1].values:  # Check second column (B)
        return prompt2
    else:
        return "No student prompt found"  # Default if tokenid is not found

def find_less_sustained_reading_examples(file_path):
    '''
    removes empty pretext0 and filters repeating examples
    '''

    with open(file_path, "r") as fh:
        data = json.load(fh)

    examples = {}

    i = 2
    while i < len(data):
        interval0 = data[i]

        if "sustained_reading_fixations" in interval0 and interval0["sustained_reading_fixations"] > 0:
            j = i + 1
            while j < len(data):
                interval1 = data[j]
                if not interval1["pretext"].startswith(interval0["pretext"]):
                    break
                if interval1["pretext"] and interval1["pretext"][-1] in ".?!":
                    pretextn = interval1["pretext"]
                    ending_id = j
                    pretext0 = interval0["pretext"].strip()  # Remove leading/trailing spaces

                    # Only store if pretext0 is non-empty
                    if pretext0:
                        # If this (pretextn, ending_id) exists, keep the lowest starting_id
                        if (pretextn, ending_id) not in examples or examples[(pretextn, ending_id)]["starting_id"] > i:
                            examples[(pretextn, ending_id)] = {
                                "starting_id": i,
                                "ending_id": j,
                                "pretext0": pretext0,
                                "pretextn": pretextn
                            }
                    break
                j += 1
        i += 1

    return list(examples.values())  # Convert dictionary to list

def get_fixations(file_path, examples, n):
    '''
    Gets fixated significant words and sentences for a given example. Also gets student prompt for the file.
    Returns the results for further analysis.
    '''
    with open(file_path, "r") as fh:
        data = json.load(fh)

    results = []

    for example in examples:
        starting_id = example["starting_id"]
        ending_id = example["ending_id"]  # Exclude this index
        starting_global_offset = data[starting_id].get("global_offset", float("inf"))

        fixated_words = []
        fixated_sentences = {}

        for i in range(starting_id, ending_id):
            entry = data[i]
            if "eye" in entry:
                for eye_data in entry["eye"]:
                    fixation_offset = eye_data.get("fixation_offset", float("inf"))

                    if fixation_offset < starting_global_offset:
                        fixated_word = eye_data.get("fixated_word")
                        fixated_sentence = eye_data.get("fixated_sentence")

                        if fixated_word:
                            cleaned_word = clean_text(fixated_word)
                            if cleaned_word.lower() not in stop_words:
                                fixated_words.append(cleaned_word)

                                if fixated_sentence:
                                    marker_index = fixated_sentence.find("<-->")
                                    if marker_index != -1:
                                        tail = fixated_sentence[marker_index + 4:]
                                        actual_tail_length = len(tail)
                                        allowed_tail_length = max(0, starting_global_offset - fixation_offset)

                                        if actual_tail_length > allowed_tail_length:
                                            fixated_sentence = fixated_sentence[:marker_index + 4 + allowed_tail_length]

                                    cleaned_sentence = clean_text(fixated_sentence)
                                    if cleaned_sentence not in fixated_sentences:
                                        fixated_sentences[cleaned_sentence] = set()
                                    fixated_sentences[cleaned_sentence].add(cleaned_word)

        # Keep only sentences with at least 3 significant fixated words
        filtered_fixated_sentences = [
            sentence for sentence, words in fixated_sentences.items()
            if len(words) >= 3 and sentence.strip() != ''
        ]

        unique_fixated_words = [word for word in set(fixated_words) if word.strip() != '']

        pretext0_words = set(
            clean_text(example["pretext0"]).split()
        )
        pretext0_words = set(
            word for word in pretext0_words if word.lower() not in stop_words
        )
        non_fixated_words = list(pretext0_words - set(unique_fixated_words))

        if len(non_fixated_words) > len(unique_fixated_words):
            non_fixated_words = random.sample(non_fixated_words, len(unique_fixated_words))

        if unique_fixated_words and filtered_fixated_sentences:
            results.append({
                "starting_id": starting_id,
                "ending_id": ending_id,
                "pretext0": example["pretext0"],
                "pretextn": example["pretextn"],
                "fixated_words": unique_fixated_words,
                "fixated_sentences": filtered_fixated_sentences,
                "non_fixated_words": non_fixated_words
            })

    return results

def get_completion(pretext0, pretextn):
    """ Extract the continuation portion of pretextn that is not in pretext0. """
    pretextn = pretextn.lstrip()  # Ignore leading spaces in pretextn
    if pretextn.startswith(pretext0):
        return pretextn[len(pretext0):].strip()  # Remove the already present part
    else:
        # If pretext0 is not exactly at the start, find the largest overlap
        for i in range(len(pretext0)):
            if pretextn.startswith(pretext0[i:]):
                return pretextn[len(pretext0[i:]):].strip()
        return pretextn.strip()  # Ensure no leading/trailing spaces in the result
