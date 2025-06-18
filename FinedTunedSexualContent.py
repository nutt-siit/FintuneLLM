import transformers
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

print(transformers.__version__)
# Sample data
data = {
    "text": [
        "You're so hot",
        "Let's play football",
        "I want to sleep with you",
        "You're a great friend"
    ],
    "label": [1, 0, 1, 0]
}

dataset = Dataset.from_dict(data)

# Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)
tokenized = dataset.map(preprocess, batched=True)

# Training setup
args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    evaluation_strategy="no",  # or "epoch" if you add a validation set
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized
)

trainer.train()

# ====== Save fine-tuned model ======
model.save_pretrained("fine_tuned_sexual_model")
tokenizer.save_pretrained("fine_tuned_sexual_model")
print("✅ Model saved to 'fine_tuned_sexual_model' folder")
