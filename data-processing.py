import json
import re
import pandas as pd
from transformers import AutoTokenizer

# Load the JSON data
data_path = 'data.json'
with open(data_path, 'r') as file:
    data = json.load(file)

# Convert JSON to a DataFrame
df = pd.DataFrame(data)

# Clean the text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['user_input'] = df['user_input'].apply(clean_text)
df['chatbot_response'] = df['chatbot_response'].apply(clean_text)

# Tokenize the data using a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Set padding token as EOS token since GPT-2 has no pad token
tokenizer.pad_token = tokenizer.eos_token

def tokenize_text(text):
    return tokenizer.encode(text, add_special_tokens=True)

df['user_input_tokens'] = df['user_input'].apply(tokenize_text)
df['chatbot_response_tokens'] = df['chatbot_response'].apply(tokenize_text)

# Pad sequences with zeros to ensure uniform length
def pad_zero(tokens, max_length):
    return tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))

# Determine the maximum length for padding
max_length = max(df['user_input_tokens'].apply(len).max(), df['chatbot_response_tokens'].apply(len).max())

# Apply padding to token columns
df['user_input_tokens'] = df['user_input_tokens'].apply(lambda x: pad_zero(x, max_length))
df['chatbot_response_tokens'] = df['chatbot_response_tokens'].apply(lambda x: pad_zero(x, max_length))

# Save the processed data for training
processed_data_path = 'processed_data.csv'
df.to_csv(processed_data_path, index=False)

print(f"Data has been preprocessed and saved to {processed_data_path}")
