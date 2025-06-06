import openai
import os

os.environ["OPENAI_API_KEY"] = "#####"

def calculate_max_tokens(student_completion, base_tokens=25):
    """
    Calculates max tokens dynamically based on the length of pretext.
    Adjust `base_tokens` as needed to account for the length of the prompt.
    """
    # Approximate tokens by dividing the number of characters by 4 (average token length in GPT-3.5).
    pretext_length = len(student_completion) // 4
    return base_tokens + pretext_length

def generate_llm_responses(student_prompt, student_completion, pretext):
    """
    Generates an LLM completion response for a single pretext.
    """

    max_tokens = calculate_max_tokens(student_completion)

    messages = [
        {
            "role": "user",
            "content": f"This was the task description provided to a student: {student_prompt}. \n"
                       f"Please write a continuation of: {pretext}"
        }
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens  # Use dynamic max_tokens
        )
        output = response['choices'][0]['message']['content']
        output = output.replace("Continuation:", "").strip()
        return output
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: {e}"

def generate_llm_responses_with_fixated_words(student_prompt, student_completion, pretext, fixations):
    """
    Generates an LLM completion response for a single pretext, considering fixated words.
    """

    max_tokens = calculate_max_tokens(student_completion)

    messages = [
        {
            "role": "user",
            "content": f"This was the task description provided to a student: {student_prompt}. \n"
                       f"We have identified the following key words as particularly important: {fixations} \n"
                       f"Please write a continuation of: {pretext}"
        }
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens  # Use dynamic max_tokens
        )
        output = response['choices'][0]['message']['content']
        output = output.replace("Continuation:", "").strip()

        return output
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: {e}"


def generate_llm_responses_with_fixated_sentences(student_prompt, student_completion, pretext, fixations):
    """
    Generates an LLM completion response for a single pretext, considering fixated words.
    """

    max_tokens = calculate_max_tokens(student_completion)

    messages = [
        {
            "role": "user",
            "content": f"This was the task description provided to a student: {student_prompt}. \n"
                       f"We have identified the following sentences as particularly important: {fixations} \n"
                       f"Please write a continuation of: {pretext}"
        }
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens  # Use dynamic max_tokens
        )
        output = response['choices'][0]['message']['content']
        output = output.replace("Continuation:", "").strip()
        return output
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: {e}"

def generate_llm_responses_control(student_prompt, student_completion, pretext, fixations):
    """
    Generates an LLM completion response for a single pretext, considering fixated words.
    """

    max_tokens = calculate_max_tokens(student_completion)

    # Prepare the message (prompt)
    messages = [
        {       "role":"user",
                "content": f"This was the task description provided to a student: {student_prompt}. \n"
                           f"We have identified the following key words as particularly important: {fixations} \n"
                           f"Please write a continuation of: {pretext}"
        }
    ]

    try:
        # Request completion from the LLM
        response = openai.ChatCompletion.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens  # Use dynamic max_tokens
        )
        # Return both the generated response and the prompt
        output = response['choices'][0]['message']['content']
        output = output.replace("Continuation:", "").strip()
        return output
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: {e}", messages[0]['content']
