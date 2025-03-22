import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import argparse

# 1. Load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

# 2. Preprocessed dataset
def preprocess_data(df):
    # Separate features and label
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols]  # Only use feature columns
    y = df['label']
    print(X.info())
    
    # Encoding label
    unique_labels = y.unique()
    if len(unique_labels) == 1:  # If there's only one label
        print(f"Warning: Dataset only has one label: {unique_labels[0]}")
        if 'negatif' in unique_labels:
            label_map = {'negatif': 0, 'positif': 1, 'netral': 2}
        else:
            label_map = {unique_labels[0]: 0, 'other': 1}
    else:
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
    
    y = y.map(label_map)
    
    # Standardize numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Combine scaled features with label
    processed_df = X_scaled_df.copy()
    processed_df['label'] = y.values
    
    # Add full_text column if it exists in original df
    if 'full_text' in df.columns:
        processed_df['full_text'] = df['full_text']
    
    return processed_df, scaler, label_map

# 3. Split data
def split_data(df, test_size=0.2, val_size=0.1):
    # If dataset is very small, use stratification only when possible
    if len(df) > 10 and len(df['label'].unique()) > 1:
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), stratify=train_df['label'], random_state=42)
    else:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), random_state=42)
    
    return train_df, val_df, test_df

# 4. Custom model combining IndobERT with numeric features
class IndobertWithNumericFeatures(nn.Module):
    def __init__(self, bert_model_name="indobenchmark/indobert-base-p1", num_features=100, num_labels=2):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_labels)
        self.num_labels = num_labels
        
        # Layer for numeric features
        self.numeric_projection = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
        
        # Layer to combine BERT output with numeric features
        self.classifier = nn.Linear(768 + 128, num_labels)
        
    def forward(self, input_ids=None, attention_mask=None, numeric_features=None, labels=None):
        # Process text with BERT
        bert_outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_outputs.pooler_output
        
        # Process numeric features
        numeric_output = self.numeric_projection(numeric_features)
        
        # Combine BERT output with numeric features
        combined_output = torch.cat([pooled_output, numeric_output], dim=1)
        
        # Classifier
        logits = self.classifier(combined_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# 5. Custom Dataset for combining numeric features with text
class TextWithNumericFeaturesDataset(TorchDataset):
    def __init__(self, df, tokenizer, text_column="full_text", max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_length = max_length
        
        # Get all feature columns
        self.feature_cols = [col for col in df.columns if col.startswith('feature_')]
        
        # Separate numeric features and label
        self.numeric_features = df[self.feature_cols].values
        self.labels = df['label'].values
        
        # If there's no text column, use placeholder
        if text_column not in df.columns:
            self.texts = ["placeholder text"] * len(df)
        else:
            self.texts = df[text_column].fillna("").tolist()  # Handle NaN values

        print(df.head())
        
        # Add column_names attribute that Trainer expects
        self.column_names = df.columns.tolist()
        
        # Define features that will be used by the model
        self.features = ['input_ids', 'attention_mask', 'numeric_features', 'labels']
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Tokenize text
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Flatten tensor for input_ids and attention_mask
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        # Get numeric features and label
        numeric_features = torch.tensor(self.numeric_features[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'numeric_features': numeric_features,
            'labels': label
        }
    
    # Add this method to handle the removal of unused columns
    def remove_columns(self, column_names):
        # This is a no-op since we're not actually removing columns
        # We just need this method to satisfy the Trainer's interface
        return self

# 6. Custom Trainer for our model
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            numeric_features=inputs['numeric_features'],
            labels=inputs['labels']
        )
        
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss
    
    # Override _remove_unused_columns to work with our custom dataset
    def _remove_unused_columns(self, dataset, description=None):
        if not isinstance(dataset, TextWithNumericFeaturesDataset):
            return super()._remove_unused_columns(dataset, description)
        
        # For our custom dataset, just return it as is
        return dataset

# 7. Metrics calculation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Accuracy
    accuracy = (preds == labels).mean()
    
    return {
        'accuracy': accuracy,
    }

# 8. Fine-tuning function
def fine_tune_indobert_with_numeric(train_df, val_df, test_df, num_features, num_labels=2, output_dir="./indobert_numeric_model"):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    
    # Create datasets
    train_dataset = TextWithNumericFeaturesDataset(train_df, tokenizer)
    val_dataset = TextWithNumericFeaturesDataset(val_df, tokenizer)
    test_dataset = TextWithNumericFeaturesDataset(test_df, tokenizer)
    
    # Initialize model
    model = IndobertWithNumericFeatures(num_features=num_features, num_labels=num_labels)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        fp16=torch.cuda.is_available()  # Use mixed precision if GPU available
    )
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Fine-tune model
    print("Starting model training...")
    trainer.train()
    
    # Evaluate model
    print("Evaluating model on test set...")
    eval_results = trainer.evaluate(test_dataset)
    print(f"Evaluation results: {eval_results}")
    
    # Save model and tokenizer
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer, eval_results

# 9. Function to predict with the fine-tuned model
def predict(model, tokenizer, scaler, features, text="", label_map=None):
    # Check if feature count matches what the scaler expects
    if len(features) != scaler.n_features_in_:
        raise ValueError(f"Expected {scaler.n_features_in_} features, but got {len(features)}")
    
    # Standardize numeric features
    scaled_features = scaler.transform([features])[0]
    scaled_features_tensor = torch.tensor(scaled_features, dtype=torch.float).unsqueeze(0)
    
    # Tokenize text
    encoding = tokenizer(
        text if text else "placeholder text",
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors="pt"
    )
    
    # Move tensors to the same device as model
    device = next(model.parameters()).device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    scaled_features_tensor = scaled_features_tensor.to(device)
    
    # Prediction
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            numeric_features=scaled_features_tensor
        )
    
    # Get logits and convert to probabilities
    logits = outputs['logits']
    probs = torch.softmax(logits, dim=1)
    
    # Get class with highest probability
    predicted_class = torch.argmax(probs, dim=1).item()
    
    # Reverse label mapping
    if label_map:
        reverse_label_map = {v: k for k, v in label_map.items()}
        predicted_label = reverse_label_map.get(predicted_class, f"Unknown class {predicted_class}")
    else:
        # Default mapping if not provided
        default_map = {0: 'negatif', 1: 'positif', 2: 'netral'}
        predicted_label = default_map.get(predicted_class, f"Unknown class {predicted_class}")
    
    return {
        'predicted_label': predicted_label,
        'confidence': probs[0][predicted_class].item(),
        'all_probabilities': {i: p.item() for i, p in enumerate(probs[0])}
    }

# 10. Main function for fine-tuning and testing
def main(file_path, output_dir="./indobert_numeric_model"):
    # Load dataset
    print(f"Loading dataset from {file_path}")
    df = load_dataset(file_path)
    
    # Make sure dataset has label column
    if 'label' not in df.columns:
        raise ValueError("Dataset must have a 'label' column")
    
    # Check feature columns
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found. Feature columns should start with 'feature_'")
    
    # Check if text column exists
    if 'full_text' not in df.columns:
        print("Warning: 'full_text' column not found. Adding placeholder text.")
        df['full_text'] = "placeholder text"
    
    # Check dataset size
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Preprocessing
    print("Preprocessing dataset...")
    processed_df, scaler, label_map = preprocess_data(df)
    num_features = len(feature_cols)  # Number of features (excluding label column)
    num_labels = len(label_map)
    
    print(f"Number of features: {num_features}")
    print(f"Number of labels: {num_labels}")
    print(f"Label mapping: {label_map}")
    
    # Split data
    print("Splitting dataset into train, validation, and test sets...")
    train_df, val_df, test_df = split_data(processed_df)
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Fine-tuning
    print(f"Fine-tuning model with {num_features} features and {num_labels} labels...")
    model, tokenizer, eval_results = fine_tune_indobert_with_numeric(
        train_df, val_df, test_df, num_features, num_labels, output_dir
    )
    
    # Save scaler and label_map for later use
    joblib.dump(scaler, os.path.join(output_dir, 'feature_scaler.pkl'))
    joblib.dump(label_map, os.path.join(output_dir, 'label_map.pkl'))
    
    print("Fine-tuning completed!")
    print(f"Model, tokenizer, scaler, and label map saved to {output_dir}")
    
    # Test prediction
    print("\nTesting prediction functionality...")
    # Get sample from test set
    sample_idx = 0
    # Get only feature columns for prediction
    sample_features = test_df.iloc[sample_idx][feature_cols].values
    actual_label_idx = test_df['label'].iloc[sample_idx]
    sample_text = ""
    if 'full_text' in test_df.columns:
        sample_text = test_df['full_text'].iloc[sample_idx]
    
    # Reverse label mapping to get actual label
    reverse_label_map = {v: k for k, v in label_map.items()}
    actual_label = reverse_label_map[actual_label_idx]
    
    # Make prediction
    prediction_result = predict(model, tokenizer, scaler, sample_features, text=sample_text, label_map=label_map)
    
    print(f"Sample features shape: {len(sample_features)}")
    print(f"Actual label: {actual_label}")
    print(f"Predicted label: {prediction_result['predicted_label']}")
    print(f"Confidence: {prediction_result['confidence']:.4f}")
    print(f"All probabilities: {prediction_result['all_probabilities']}")
    
    # Load model for demo
    print("\nTesting model loading and prediction...")
    loaded_model = IndobertWithNumericFeatures(num_features=num_features, num_labels=num_labels)
    loaded_model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
    loaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    loaded_scaler = joblib.load(os.path.join(output_dir, 'feature_scaler.pkl'))
    loaded_label_map = joblib.load(os.path.join(output_dir, 'label_map.pkl'))
    
    # Predict with loaded model
    loaded_prediction = predict(
        loaded_model, loaded_tokenizer, loaded_scaler, 
        sample_features, text=sample_text, label_map=loaded_label_map
    )
    
    print(f"Loaded model prediction: {loaded_prediction['predicted_label']}")
    print(f"Loaded model confidence: {loaded_prediction['confidence']:.4f}")
    
    print("\nFine-tuning and testing completed successfully!")

# 11. Example usage with test data
def example_usage():
    # Load fine-tuned model and tokenizer
    model_dir = "./indobert_numeric_model"
    
    # Check if model exists
    if not os.path.exists(os.path.join(model_dir, 'pytorch_model.bin')):
        print(f"Model not found at {model_dir}")
        return
    
    # Load model components
    loaded_scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
    loaded_label_map = joblib.load(os.path.join(model_dir, 'label_map.pkl'))
    
    # Get the number of features from the scaler
    num_features = loaded_scaler.n_features_in_
    num_labels = len(loaded_label_map)
    
    print(f"Model expects {num_features} features and {num_labels} labels")
    
    loaded_model = IndobertWithNumericFeatures(num_features=num_features, num_labels=num_labels)
    loaded_model.load_state_dict(torch.load(os.path.join(model_dir, 'pytorch_model.bin')))
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Ensure we have the correct number of features
    # Example data for prediction (feature values from feature_0 to feature_99)
    sample_features = [0.002143046927529496, 0.26623434704039006, 0.167149575112702, -0.05220156626219486, 
                     0.2189711079896464, 0.018437574182141137, -0.16046413748718605, 0.36677507935212805, 
                     -0.024257547657501388, -0.24033706375192126, 0.06668521737220226, -0.23177560389815988, 
                     -0.0019846331062061445, -0.06042115648531101, 0.24804522821607142, 0.07294832218994761, 
                     0.21057214459718823, 0.32351112559244233, -0.1620595252939633, -0.12053641543856689, 
                     -0.05431940669629287, -0.1028457141493435, 0.2362318696067124, 0.04523195843998488, 
                     -0.02531634271144867, 0.3085106764133875, 0.10432410883632573, 0.14470700680145196, 
                     0.0591323964733195, 0.07288066670298576, 0.10839007594078393, -0.19656206500510892, 
                     -0.1923886083743789, -0.05932130359790542, -0.03605488951426822, -0.05038791255082016, 
                     0.16156265110551538, -0.016968070202833647, -0.2581631426590604, 0.03850175690679968, 
                     0.11186220769032643, 0.14592142083815166, -0.039984622546894985, 0.19060918737154503, 
                     0.00964412179037735, 0.18804129793659433, 0.06515240098664223, -0.18301488369606533, 
                     -0.31349791170327695, -0.06304637347529461, 0.12692525063629273, -0.22180984928816944, 
                     -0.15856407398914363, -0.24048959792519062, -0.33421095226008396, -0.10709329150683645, 
                     -0.3454290961048433, -0.024069625194873902, 0.05195168364647921, -0.07670715379879459, 
                     0.03434324368544213, 0.3398816351833233, 0.14443038127010133, 0.1376984960508424, 
                     0.09868173354438373, -0.16845776757539868, 0.14822543118233708, 0.08091714963703961, 
                     0.3131577599923695, -0.113702052705256, 0.13311533385573268, -0.011846642360671775, 
                     -0.14327624875616718, 0.23477030516444863, -0.20303775065324522, 0.21105906311664488, 
                     -0.1697741864708724, -0.2752409722873723, 0.02634545062463005, 0.09325183998769174, 
                     0.2890492825248799, -0.03187524807917607, -0.03759083202616735, -0.2856282689548158, 
                     -0.11075184215464956, 0.08499520011367155, -0.14643176833724048, -0.1776084879884956, 
                     -0.0793930504519444, -0.2761946776518961, -0.19352826933594203, -0.019309434265672387, 
                     -0.08957609863250286, 0.11952889313945522, -0.2243741312874602, 0.030825556775966247, 
                     0.04832623103087732, -0.046741928360969215, -0.03780248395905092, -0.14539733215827833]
    
    # Make sure sample_features has the right length
    if len(sample_features) != num_features:
        print(f"Warning: Sample features has {len(sample_features)} elements but model expects {num_features}")
        # Adjust the features array to match expected size
        if len(sample_features) < num_features:
            # Add zeros if we have too few
            sample_features = sample_features + [0.0] * (num_features - len(sample_features))
        else:
            # Truncate if we have too many
            sample_features = sample_features[:num_features]
    
    # Sample text for prediction
    sample_text = "This is a sample text for prediction."
    
    # Make prediction
    prediction = predict(loaded_model, loaded_tokenizer, loaded_scaler, sample_features, 
                         text=sample_text, label_map=loaded_label_map)
    
    print("\nPrediction result:")
    print(f"Predicted label: {prediction['predicted_label']}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print(f"All probabilities: {prediction['all_probabilities']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune IndobERT with numeric features')
    parser.add_argument('--data_path', type=str, default='./preprocess_data/word2vec_vectors(withlabel).csv', help='Path to the dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='./indobert_numeric_model', help='Output directory to save model')
    parser.add_argument('--example', action='store_true', help='Run example usage with a pre-trained model')
    
    args = parser.parse_args()
    
    if args.example:
        example_usage()
    else:
        main(args.data_path, args.output_dir)