## Overview

This repository contains the source code used in the paper:

**EyeLLM: Using Lookback Fixations to Enhance Human-LLM Alignment for Text Completion**  
*Astha Singh, Mark Torrance, Evgeny Chukharev*  

---

## Installation

Step-by-step instructions to set up the environment.

```bash
# Clone the repository
git clone https://github.com/astha99/EyeLLM.git
cd EyeLLM

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

src/main: Includes the source code files for getting hesitation events and generating LLM responses

1. dataWrangling.py - Contains functions that extract valid hesitation events and eye fixations (words, sentences and non-fixated words)
2. similarityScores.py - Contains functions to calculate different similarity scores (Semantic, F1, Jaccard)
3. GPTresponseGenerator.py - Controls LLM settings for GPT models (gpt-3.5-turbo, gpt-4.1)
4. responseGenerator.py - Uses Ollama to control response generation settings for Mistral7B and LLaMa3-8B
5. compiled.py - Script to generate and save responses for GPT models
6. nonGPTcompiled.py - Script to generate and save responses for Mistral7B and LLaMa3

src/post-processing:
