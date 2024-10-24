import json
import re
import pandas as pd
from transformers import AutoTokenizer, DataCollatorWithPadding, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

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

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Explicitly add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# Load preprocessed data
data_path = 'processed_data.csv'
df = pd.read_csv(data_path)

# Create a custom dataset class
class FitnessChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.inputs = data['user_input'].values
        self.responses = data['chatbot_response'].values
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        response_text = self.responses[idx]
        combined_text = input_text + " " + self.tokenizer.eos_token + " " + response_text

        # Encode with padding to the max_length
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Flatten the tensors and return as a dictionary
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()  # Labels are the same as input_ids for language modeling
        }

# Split the data into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create dataset instances
train_dataset = FitnessChatDataset(train_df, tokenizer)
test_dataset = FitnessChatDataset(test_df, tokenizer)

# Use DataCollatorWithPadding for handling varying sequence lengths
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Set device to MPS (Apple Silicon GPU) if available, otherwise use CPU
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
model.to(device)

# Set updated training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,  # Increase the number of epochs if desired
    per_device_train_batch_size=4,  # Increase batch size if hardware allows
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=3e-5,  # Lower the learning rate for better convergence
    weight_decay=0.05,  # Experiment with weight decay to control overfitting
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    eval_steps=500,
    max_grad_norm=1.0  # Clip gradients to avoid large updates
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./results')
tokenizer.save_pretrained('./results')

# Evaluate the model on the test dataset
print("*************************************Evaluating Model**********************************")
eval_results = trainer.evaluate(eval_dataset=test_dataset)
print("Evaluation results:", eval_results)