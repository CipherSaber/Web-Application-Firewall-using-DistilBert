import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import evaluate
from sklearn.model_selection import train_test_split

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# UPDATE THIS PATH TO YOUR ACTUAL CSV FILE
csv_path = 'csic_database.csv'  # Replace with your actual CSV file path
df = pd.read_csv(csv_path)

print("Initial dataframe shape:", df.shape)
print("Columns:", df.columns)

# Check classification column type and values
print("Classification dtype:", df['classification'].dtype)
print("Unique classification values:", df['classification'].unique())
print("Classification value counts:", df['classification'].value_counts())

# Handle both string and numeric labels
if df['classification'].dtype == 'object':
    # String labels - convert to lowercase
    df['classification'] = df['classification'].str.lower()
    label_map = {'normal': 0, 'anomalous': 1}
else:
    # Numeric labels - use as is or map appropriately
    label_map = {0: 0, 1: 1}  # assuming 0=normal, 1=anomalous

df['label'] = df['classification'].map(label_map)
df = df.dropna(subset=['label'])

# Combine HTTP request parts into single request string
def combine_request(row):
    method = str(row['Method']) if pd.notna(row['Method']) else ""
    url = str(row['URL']) if pd.notna(row['URL']) else ""
    user_agent = str(row['User-Agent']) if pd.notna(row['User-Agent']) else ""
    return f"{method} {url} {user_agent}"

df['request'] = df.apply(combine_request, axis=1)
df_clean = df.dropna(subset=['request', 'label'])

print("Cleaned dataframe shape:", df_clean.shape)
if df_clean.empty:
    raise ValueError("No data left after cleaning. Check your dataset and preprocessing!")

# Train/validation split
train_df, val_df = train_test_split(
    df_clean[['request', 'label']],
    test_size=0.2,
    random_state=42,
    stratify=df_clean['label']
)

print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def preprocess(examples):
    return tokenizer(examples['request'], truncation=True, padding='max_length', max_length=128)

# Prepare datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

train_dataset = train_dataset.map(preprocess, batched=True)
val_dataset = val_dataset.map(preprocess, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Load model and move to GPU
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.to(device)

accuracy_metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

# Training arguments optimized for RTX 3050 laptop (4GB VRAM)
training_args = TrainingArguments(
    output_dir='./model_checkpoint',
    do_train=True,
    do_eval=True,
    eval_strategy='steps',
    eval_steps=100,
    save_steps=500,
    save_total_limit=2,
    num_train_epochs=4,
    per_device_train_batch_size=16,  # Optimized for 4GB VRAM
    per_device_eval_batch_size=16,   # Optimized for 4GB VRAM
    gradient_accumulation_steps=4,   # Effective batch size = 16 * 4 = 64
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    remove_unused_columns=False,
    dataloader_pin_memory=True,      # Speed up data loading
    fp16=True,                       # Mixed precision for memory savings
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

if __name__ == '__main__':
    print("Starting GPU training...")
    trainer.train()
    
    print("Saving model and tokenizer...")
    trainer.save_model('./model_checkpoint')
    tokenizer.save_pretrained('./model_checkpoint')
    
    print("Training completed!")
