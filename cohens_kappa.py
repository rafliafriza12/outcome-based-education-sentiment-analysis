import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from collections import Counter

# Function to load data from CSV files
def load_data_from_csv(manual_file_path, gpt_file_path):
    # Load CSV files
    manual_df = pd.read_csv(manual_file_path)
    gpt_df = pd.read_csv(gpt_file_path)
    
    # Check that both dataframes have the same texts
    if not manual_df['full_text'].equals(gpt_df['full_text']):
        print("Warning: The texts in the two CSV files don't match exactly.")
        
    return manual_df, gpt_df

# Calculate Cohen's Kappa
def calculate_cohen_kappa(manual_df, gpt_df):
    # Extract the labels (annotations)
    manual_annotations = manual_df['label'].tolist()
    gpt_annotations = gpt_df['label'].tolist()
    
    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(manual_annotations, gpt_annotations)
    
    return kappa, manual_annotations, gpt_annotations

# Create a confusion matrix
def create_confusion_matrix(y_true, y_pred):
    labels = sorted(list(set(y_true + y_pred)))
    n_labels = len(labels)
    
    # Create a dictionary mapping label to index
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # Initialize confusion matrix
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    # Fill confusion matrix
    for true, pred in zip(y_true, y_pred):
        cm[label_to_idx[true], label_to_idx[pred]] += 1
    
    return cm, labels

# Calculate agreement metrics per class
def per_class_metrics(cm, labels):
    n_labels = len(labels)
    metrics = {}
    
    for i, label in enumerate(labels):
        true_pos = cm[i, i]
        false_pos = sum(cm[:, i]) - true_pos
        false_neg = sum(cm[i, :]) - true_pos
        
        # Count instances where both annotators agree this is not the class
        true_neg = np.sum(cm) - true_pos - false_pos - false_neg
        
        # Calculate agreement percentage for this class
        total = np.sum(cm)
        agreement = (true_pos + true_neg) / total
        
        metrics[label] = {
            "agreement": agreement,
            "true_positive": true_pos,
            "false_positive": false_pos,
            "false_negative": false_neg,
            "true_negative": true_neg
        }
    
    return metrics

# Interpret Cohen's Kappa
def interpret_kappa(kappa):
    if kappa < 0:
        return "Poor agreement (less than chance)"
    elif kappa < 0.2:
        return "Slight agreement"
    elif kappa < 0.4:
        return "Fair agreement"
    elif kappa < 0.6:
        return "Moderate agreement"
    elif kappa < 0.8:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"

def main():
    # File paths for the CSV files
    manual_file_path = "./labeled_data/labeled_sentiment_manual_annotation.csv"  # Replace with your actual file path
    gpt_file_path = "./labeled_data/labeled_sentiment_gpt_for_cohens_kappa.csv"        # Replace with your actual file path
    
    # Load data from CSV files
    manual_df, gpt_df = load_data_from_csv(manual_file_path, gpt_file_path)
    
    # Calculate Cohen's Kappa
    kappa, manual_annotations, gpt_annotations = calculate_cohen_kappa(manual_df, gpt_df)
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Kappa Interpretation: {interpret_kappa(kappa)}")
    
    # Create and display confusion matrix
    cm, labels = create_confusion_matrix(manual_annotations, gpt_annotations)
    print("\nConfusion Matrix:")
    print("Labels:", labels)
    print(cm)
    
    # Calculate and display per-class metrics
    class_metrics = per_class_metrics(cm, labels)
    print("\nPer-Class Metrics:")
    for label, metrics in class_metrics.items():
        print(f"\nClass: {label}")
        print(f"Agreement: {metrics['agreement']:.4f}")
        print(f"True Positives: {metrics['true_positive']}")
        print(f"False Positives: {metrics['false_positive']}")
        print(f"False Negatives: {metrics['false_negative']}")
        print(f"True Negatives: {metrics['true_negative']}")
    
    # Calculate overall percentage agreement
    correct = sum(1 for m, g in zip(manual_annotations, gpt_annotations) if m == g)
    total = len(manual_annotations)
    percentage_agreement = correct / total * 100
    print(f"\nOverall Percentage Agreement: {percentage_agreement:.2f}%")
    
    # Display distribution of annotations
    print("\nDistribution of Manual Annotations:")
    print(Counter(manual_annotations))
    print("\nDistribution of GPT-4o-mini Annotations:")
    print(Counter(gpt_annotations))

if __name__ == "__main__":
    main()