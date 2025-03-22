import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. Load data
def load_data(file_path):
    # Membaca file CSV dengan format yang sesuai
    df = pd.read_csv(file_path)
    
    # Ekstrak fitur dan label
    X = df.iloc[:, :-2]  # Semua kolom kecuali kolom terakhir (label)
    y = df.iloc[:, -1]   # Kolom terakhir adalah label
    
    return X, y

# 2. Split data menjadi training dan testing
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

# 3. Train model SVM dengan SMOTE dan hyperparameter tuning
def train_svm_model_with_smote(X_train, y_train):
    # Pipeline dengan SMOTE, scaling, dan SVM
    smote_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42))
    ])
    
    # Parameter grid untuk grid search
    param_grid = {
        'smote__k_neighbors': [5, 10],  # Jumlah tetangga untuk SMOTE
        'svm__C': [0.1, 1, 10],
        'svm__gamma': ['scale', 'auto', 0.1],
        'svm__kernel': ['rbf', 'linear'],
        'svm__class_weight': ['balanced', None]
    }
    
    # Grid search untuk menemukan parameter terbaik
    grid_search = GridSearchCV(
        smote_pipeline, param_grid, cv=5, scoring='balanced_accuracy', verbose=1, n_jobs=-1
    )
    
    print("Memulai grid search untuk SVM dengan SMOTE...")
    # Fit model
    grid_search.fit(X_train, y_train)
    
    # Tampilkan parameter terbaik
    print("Parameter terbaik: ", grid_search.best_params_)
    print(f"Balanced accuracy terbaik dari cross-validation: {grid_search.best_score_:.4f}")
    
    # Return model dengan parameter terbaik
    return grid_search.best_estimator_

# 4. Evaluasi model
def evaluate_model(model, X_test, y_test):
    # Prediksi pada data test
    y_pred = model.predict(X_test)
    
    # Hitung akurasi
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"Akurasi: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    # Tampilkan classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(y_test.unique()), 
                yticklabels=sorted(y_test.unique()))
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title('Confusion Matrix - SVM dengan SMOTE')
    plt.show()
    
    return accuracy, y_pred

# 5. Visualisasi distribusi label sebelum dan sesudah SMOTE
def plot_class_distribution(y_train, y_resampled):
    plt.figure(figsize=(12, 5))
    
    # Distribusi sebelum SMOTE
    plt.subplot(1, 2, 1)
    train_counts = pd.Series(y_train).value_counts()
    bars = sns.countplot(x=y_train)
    # Tambahkan label jumlah di atas bar
    for i, count in enumerate(train_counts):
        bars.text(i, count + 5, str(count), ha='center')
    plt.title('Distribusi Kelas Sebelum SMOTE')
    plt.xlabel('Label')
    plt.ylabel('Jumlah')
    
    # Distribusi setelah SMOTE
    plt.subplot(1, 2, 2)
    resampled_counts = pd.Series(y_resampled).value_counts()
    bars = sns.countplot(x=y_resampled)
    # Tambahkan label jumlah di atas bar
    for i, count in enumerate(resampled_counts):
        bars.text(i, count + 5, str(count), ha='center')
    plt.title('Distribusi Kelas Setelah SMOTE')
    plt.xlabel('Label')
    plt.ylabel('Jumlah')
    
    plt.tight_layout()
    plt.show()

# 6. Prediksi sentimen untuk data baru
def predict_sentiment(model, new_data):
    # Prediksi label
    prediction = model.predict(new_data)
    
    # Prediksi probabilitas
    probabilities = model.predict_proba(new_data)
    
    return prediction, probabilities

# 7. Fungsi untuk membandingkan performa model dengan dan tanpa SMOTE
def compare_models(X_train, X_test, y_train, y_test):
    # Model SVM tanpa SMOTE
    print("\n=== Training SVM TANPA SMOTE ===")
    svm_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42, class_weight='balanced'))
    ])
    
    # Fit model tanpa SMOTE
    svm_pipeline.fit(X_train, y_train)
    
    # Evaluasi model tanpa SMOTE
    print("\nEvaluasi SVM TANPA SMOTE:")
    y_pred_no_smote = svm_pipeline.predict(X_test)
    accuracy_no_smote = accuracy_score(y_test, y_pred_no_smote)
    balanced_acc_no_smote = balanced_accuracy_score(y_test, y_pred_no_smote)
    
    print(f"Akurasi: {accuracy_no_smote:.4f}")
    print(f"Balanced Accuracy: {balanced_acc_no_smote:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_no_smote))
    
    # Model SVM dengan SMOTE (sederhana, tanpa grid search)
    print("\n=== Training SVM DENGAN SMOTE (Versi Sederhana) ===")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    svm_smote_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42))
    ])
    
    # Fit model dengan SMOTE
    svm_smote_pipeline.fit(X_train_smote, y_train_smote)
    
    # Evaluasi model dengan SMOTE
    print("\nEvaluasi SVM DENGAN SMOTE:")
    y_pred_smote = svm_smote_pipeline.predict(X_test)
    accuracy_smote = accuracy_score(y_test, y_pred_smote)
    balanced_acc_smote = balanced_accuracy_score(y_test, y_pred_smote)
    
    print(f"Akurasi: {accuracy_smote:.4f}")
    print(f"Balanced Accuracy: {balanced_acc_smote:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_smote))
    
    # Plot perbandingan
    labels = ['SVM Tanpa SMOTE', 'SVM Dengan SMOTE']
    accuracies = [accuracy_no_smote, accuracy_smote]
    balanced_accs = [balanced_acc_no_smote, balanced_acc_smote]
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(labels, accuracies, color=['blue', 'green'])
    plt.ylim(0, 1)
    plt.title('Perbandingan Akurasi')
    plt.ylabel('Akurasi')
    
    plt.subplot(1, 2, 2)
    plt.bar(labels, balanced_accs, color=['blue', 'green'])
    plt.ylim(0, 1)
    plt.title('Perbandingan Balanced Accuracy')
    plt.ylabel('Balanced Accuracy')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'no_smote': {'accuracy': accuracy_no_smote, 'balanced_acc': balanced_acc_no_smote},
        'smote': {'accuracy': accuracy_smote, 'balanced_acc': balanced_acc_smote}
    }

# 8. Fungsi utama
def main(file_path):
    # Load data
    print("Loading data...")
    X, y = load_data(file_path)
    print(f"Data shape: {X.shape}")
    
    # Informasi kelas
    print("\nDistribusi label original:")
    class_counts = y.value_counts()
    print(class_counts)
    print(f"Rasio label: {class_counts.values[0] / class_counts.values[-1]:.2f}:1")
    
    # Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Terapkan SMOTE hanya untuk demo visualisasi
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print("\nDistribusi setelah SMOTE:")
    resampled_counts = pd.Series(y_resampled).value_counts()
    print(resampled_counts)
    print(f"Rasio label setelah SMOTE: {resampled_counts.values[0] / resampled_counts.values[-1]:.2f}:1")
    
    # Visualisasi distribusi sebelum dan sesudah SMOTE
    plot_class_distribution(y_train, y_resampled)
    
    # Perbandingan model dengan dan tanpa SMOTE (opsional)
    print("\nMembandingkan model SVM dengan dan tanpa SMOTE...")
    results = compare_models(X_train, X_test, y_train, y_test)
    
    # Train model SVM dengan SMOTE dan hyperparameter tuning
    print("\nTraining SVM model dengan SMOTE dan Grid Search...")
    model = train_svm_model_with_smote(X_train, y_train)
    
    # Evaluasi model final
    print("\nEvaluasi model final pada data test:")
    evaluate_model(model, X_test, y_test)
    
    # Simpan model (opsional)
    # from joblib import dump
    # dump(model, 'sentiment_svm_smote_model.joblib')
    
    return model

# Jalankan program
if __name__ == "__main__":
    file_path = "./preprocess_data/word2vec_vectors(withlabel).csv"  # Ganti dengan path file Anda
    model = main(file_path)
    
    # Contoh prediksi data baru (jika ada)
    # new_data = pd.DataFrame([...])  # Masukkan vektor word2vec dari teks baru
    # prediction, probabilities = predict_sentiment(model, new_data)
    # print(f"Prediksi: {prediction}")
    # print(f"Probabilitas: {probabilities}")