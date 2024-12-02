import openai
import pandas as pd
import os

def load_parquet_files(folder):
    """Load all Parquet files in a folder into a dictionary."""
    datasets = {}
    for file in os.listdir(folder):
        if file.endswith(".parquet"):
            category = file.replace(".parquet", "")
            datasets[category] = pd.read_parquet(os.path.join(folder, file))
    return datasets

def get_relevant_content(user_query, datasets):
    """Find the most relevant content based on the user query."""
    # For simplicity, we return a random piece of content from the relevant category
    for category, df in datasets.items(): 
        print(f"Category: {category}")
        if category in user_query.lower():
            return df["content"].sample(1).values[0]
    return None

import openai

def generate_openai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[ 
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.7,
    )
    answer = response.choices[0].message['content'].strip()
    return answer

