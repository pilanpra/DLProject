import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

# Load the trained model and tokenizer
model_path = "./results"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Explicitly add a pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Set device to MPS or CPU
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
model.to(device)
model.eval()  # Set the model to evaluation mode

# Interactive loop
print("Chat with your diet and fitness chatbot! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Tokenize user input
    inputs = tokenizer(user_input, return_tensors="pt", padding=True).to(device)

    # Generate response with adjusted parameters
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=100,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1,
            temperature=0.9,  # Increase randomness
            do_sample=True,   # Enable sampling to diversify responses
            top_k=50,         # Limit to top k tokens
            top_p=0.9         # Nucleus sampling for further control
        )

    # Decode and print the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Bot: {response}")
