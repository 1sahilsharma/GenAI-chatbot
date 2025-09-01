#!/usr/bin/env python3
# Make sure to activate the virtual environment first: source venv/bin/activate

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
# set your Groq API key

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# models = client.models.list()

# # print all model ids
# for model in models.data:
#     print(model.id)

completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {"role": "system", "content": "You are expert in Economics."},
        {"role": "user", "content": "What is Game theory?"},
    ],
)


print(completion.choices[0].message)
