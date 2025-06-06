from src.main.GPTresponseGenerator import calculate_max_tokens
import ollama

def generate_llm_responses_hf(student_prompt, student_completion, pretext, model="mistral"):
    """
    Generates an LLM completion response using a local Ollama model like mistral or llama3.
    """

    max_tokens = calculate_max_tokens(student_completion)  # Your own token function

    prompt = (
        f"This was the task description provided to a student: {student_prompt}.\n"
        f"Please write a continuation of: {pretext}"
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            options={"num_predict": max_tokens}
        )
        generated_text = response['message']['content']

        # Only remove the prompt if it's repeated at the start
        if generated_text.startswith(prompt):
            cleaned_text = generated_text[len(prompt):].strip()
        else:
            cleaned_text = generated_text.strip()

        return cleaned_text

    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: {e}", prompt


def generate_llm_responses_with_fixated_words_hf(student_prompt, student_completion, pretext, fixations, model):
    """
    Generates LLM completions using Hugging Face hosted models (like Mistral or LLaMA).
    """

    max_tokens = calculate_max_tokens(student_completion)
    prompt = (
        f"This was the task description provided to a student: {student_prompt}. \n"
                       f"We have identified the following key words as particularly important: {fixations} \n"
                       f"Please write a continuation of: {pretext}"
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            options={"num_predict": max_tokens}
        )
        generated_text = response['message']['content']

        # Only remove the prompt if it's repeated at the start
        if generated_text.startswith(prompt):
            cleaned_text = generated_text[len(prompt):].strip()
        else:
            cleaned_text = generated_text.strip()

        return cleaned_text

    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: {e}", prompt

    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: {e}"

def generate_llm_responses_with_fixated_sentences_hf(student_prompt, student_completion, pretext, fixations, model):
    """
    Generates LLM completions using Hugging Face hosted models (like Mistral or LLaMA).
    """

    max_tokens = calculate_max_tokens(student_completion)
    prompt = (
        f"This was the task description provided to a student: {student_prompt}. \n"
        f"We have identified the following sentences as particularly important: {fixations} \n"
        f"Please write a continuation of: {pretext}"
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            options={"num_predict": max_tokens}
        )
        generated_text = response['message']['content']

        # Only remove the prompt if it's repeated at the start
        if generated_text.startswith(prompt):
            cleaned_text = generated_text[len(prompt):].strip()
        else:
            cleaned_text = generated_text.strip()

        return cleaned_text

    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error: {e}", prompt