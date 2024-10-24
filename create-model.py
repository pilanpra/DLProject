from transformers import DataCollatorWithPadding, GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import pandas as pd

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

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

# Create a dataset instance
dataset = FitnessChatDataset(df, tokenizer)

# Use DataCollatorWithPadding for handling varying sequence lengths
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# Train the model
trainer.train()