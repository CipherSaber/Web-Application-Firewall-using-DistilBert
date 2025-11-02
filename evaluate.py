import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load validation data
csv_path = 'csic_database.csv'  # UPDATE THIS
df = pd.read_csv(csv_path)

print("="*60)
print("DEBUGGING DATA LOADING")
print("="*60)
print(f"Initial dataframe shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)

# Check if 'classification' column exists
if 'classification' not in df.columns:
    print("\n❌ ERROR: 'classification' column not found!")
    print(f"Available columns: {df.columns.tolist()}")
    print("\nPlease check your CSV file column names.")
    print("Looking for common alternatives...")
    
    # Try to find the label column
    possible_label_cols = ['label', 'class', 'type', 'Classification', 'Label']
    label_col = None
    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            print(f"✓ Found label column: '{col}'")
            break
    
    if label_col:
        df['classification'] = df[label_col]
    else:
        print("\n❌ Could not find label column. Please specify manually.")
        exit(1)

print(f"\nClassification column found!")
print(f"Unique values: {df['classification'].unique()}")
print(f"Value counts:\n{df['classification'].value_counts()}")

# Preprocessing - handle both string and numeric labels
if df['classification'].dtype == 'object':
    print("\nLabel type: String - converting to lowercase")
    df['classification'] = df['classification'].str.lower()
    
    # Check actual unique values after lowercase
    unique_vals = df['classification'].unique()
    print(f"Unique values after lowercase: {unique_vals}")
    
    # Create flexible label mapping
    if 'normal' in unique_vals:
        label_map = {'normal': 0, 'anomalous': 1}
    elif 'benign' in unique_vals:
        label_map = {'benign': 0, 'malicious': 1}
    elif 'legitimate' in unique_vals:
        label_map = {'legitimate': 0, 'attack': 1}
    else:
        print(f"\n❌ Unknown label format: {unique_vals}")
        print("Please specify the label mapping manually")
        exit(1)
else:
    print("\nLabel type: Numeric")
    label_map = {0: 0, 1: 1}

print(f"Using label mapping: {label_map}")
df['label'] = df['classification'].map(label_map)

print(f"\nAfter label mapping:")
print(f"Non-null labels: {df['label'].notna().sum()}")
print(f"Null labels: {df['label'].isna().sum()}")

df = df.dropna(subset=['label'])
print(f"After dropping nulls: {df.shape}")

if df.empty:
    print("\n❌ ERROR: DataFrame is empty after label mapping!")
    print("Check if your label values match the mapping.")
    exit(1)

# Combine request fields
def combine_request(row):
    method = str(row['Method']) if 'Method' in row and pd.notna(row['Method']) else ""
    url = str(row['URL']) if 'URL' in row and pd.notna(row['URL']) else ""
    user_agent = str(row['User-Agent']) if 'User-Agent' in row and pd.notna(row['User-Agent']) else ""
    
    # If columns don't exist, try alternatives
    if not method and 'method' in row:
        method = str(row['method']) if pd.notna(row['method']) else ""
    if not url and 'url' in row:
        url = str(row['url']) if pd.notna(row['url']) else ""
    if not user_agent and 'user_agent' in row:
        user_agent = str(row['user_agent']) if pd.notna(row['user_agent']) else ""
    
    return f"{method} {url} {user_agent}".strip()

df['request'] = df.apply(combine_request, axis=1)
print(f"\nSample requests:")
print(df['request'].head(3).tolist())

# Check if requests are empty
empty_requests = (df['request'] == "") | (df['request'].isna())
print(f"Empty requests: {empty_requests.sum()}")

if empty_requests.sum() == len(df):
    print("\n❌ ERROR: All requests are empty!")
    print("Available columns for request construction:", df.columns.tolist())
    exit(1)

df = df[df['request'] != ""]

print(f"\nFinal dataframe shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")

if len(df) == 0:
    print("\n❌ ERROR: No valid data after preprocessing!")
    exit(1)

print("\n" + "="*60)
print("✓ Data preprocessing successful!")
print("="*60)

# Use validation split (same as training)
_, val_df = train_test_split(
    df[['request', 'label']],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"\nValidation set size: {len(val_df)}")
print(f"Validation label distribution:\n{val_df['label'].value_counts()}")

# Load model
model_path = './model_checkpoint/checkpoint-3056'  # UPDATE THIS
print(f"\nLoading model from: {model_path}")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

print("✓ Model loaded successfully\n")
print("Starting evaluation...")

# Run predictions
predictions = []
true_labels = []

with torch.no_grad():
    for idx, row in val_df.iterrows():
        # Tokenize
        inputs = tokenizer(
            row['request'],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).cpu().item()
        
        predictions.append(pred)
        true_labels.append(int(row['label']))
        
        # Progress
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(val_df)} samples")

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, predictions, average='binary'
)

print("\n" + "="*50)
print("=== EVALUATION METRICS ===")
print("="*50)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print("="*50)

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
print("\n=== CONFUSION MATRIX ===")
print("                Predicted Normal  Predicted Anomalous")
print(f"Actual Normal        {cm[0][0]:>6}          {cm[0][1]:>6}")
print(f"Actual Anomalous     {cm[1][0]:>6}          {cm[1][1]:>6}")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Confusion Matrix Heatmap
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', 
    xticklabels=['Normal', 'Anomalous'],
    yticklabels=['Normal', 'Anomalous'],
    ax=axes[0], 
    cbar_kws={'label': 'Count'},
    annot_kws={'size': 14, 'weight': 'bold'}
)
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')

# 2. Performance Metrics Bar Chart
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy*100, precision*100, recall*100, f1*100]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

bars = axes[1].bar(metrics_names, metrics_values, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
axes[1].set_ylim([0, 100])
axes[1].set_ylabel('Score (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Performance Metrics', fontsize=14, fontweight='bold', pad=15)
axes[1].axhline(y=90, color='gray', linestyle='--', linewidth=1.5, 
                alpha=0.5, label='90% Baseline')
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
axes[1].legend(loc='lower right')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[1].text(
        bar.get_x() + bar.get_width()/2., height,
        f'{height:.2f}%',
        ha='center', va='bottom', 
        fontsize=11, fontweight='bold'
    )

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved as 'evaluation_results.png'")

# Save results to CSV
results_df = pd.DataFrame({
    'Metric': metrics_names + ['True Negatives', 'False Positives', 
                               'False Negatives', 'True Positives'],
    'Value': [accuracy*100, precision*100, recall*100, f1*100,
              cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
})
results_df.to_csv('evaluation_metrics.csv', index=False)
print(f"✓ Metrics saved to 'evaluation_metrics.csv'\n")

plt.show()
