import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
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

# 3. Train model Naive Bayes dengan SMOTE dan hyperparameter tuning
def train_nb_model_with_smote(X_train, y_train):
    # Pipeline dengan SMOTE, scaling, dan Naive Bayes
    smote_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('nb', GaussianNB())
    ])
    
    # Parameter grid untuk grid search
    param_grid = {
        'smote__k_neighbors': [5, 10, 15],  # Jumlah tetangga untuk SMOTE
        'nb__var_smoothing': np.logspace(-9, -1, 5)  # Hyperparameter var_smoothing
    }
    
    # Grid search untuk menemukan parameter terbaik
    grid_search = GridSearchCV(
        smote_pipeline, param_grid, cv=5, scoring='balanced_accuracy', verbose=1, n_jobs=-1
    )
    
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
    print(f"Akurasi pada data test: {accuracy:.4f}")
    
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
    plt.title('Confusion Matrix - Naive Bayes dengan SMOTE')
    plt.show()
    
    return accuracy, y_pred

# 5. Visualisasi distribusi label sebelum dan sesudah SMOTE
def plot_class_distribution(y_train, y_resampled):
    plt.figure(figsize=(12, 5))
    
    # Distribusi sebelum SMOTE
    plt.subplot(1, 2, 1)
    sns.countplot(x=y_train)
    plt.title('Distribusi Kelas Sebelum SMOTE')
    plt.xlabel('Label')
    plt.ylabel('Jumlah')
    
    # Distribusi setelah SMOTE
    plt.subplot(1, 2, 2)
    sns.countplot(x=y_resampled)
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

# 7. Fungsi untuk membandingkan distribusi fitur berdasarkan kelas
def plot_feature_distributions(X, y, num_features=5):
    # Gabungkan fitur dan label
    data = X.copy()
    data['label'] = y
    
    # Pilih beberapa fitur untuk visualisasi
    selected_features = X.columns[:num_features]
    
    # Plot distribusi fitur berdasarkan kelas
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(selected_features):
        plt.subplot(2, 3, i+1)
        for label in data['label'].unique():
            sns.kdeplot(data[data['label'] == label][feature], label=label)
        plt.title(f'Distribusi {feature}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

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
    
    # Visualisasi distribusi fitur (opsional)
    print("\nMenampilkan distribusi fitur berdasarkan kelas...")
    plot_feature_distributions(X, y)
    
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
    
    # Train model
    print("\nTraining Naive Bayes model dengan SMOTE dan Grid Search...")
    model = train_nb_model_with_smote(X_train, y_train)
    
    # Evaluasi model
    print("\nEvaluasi model pada data test:")
    evaluate_model(model, X_test, y_test)
    
    # Simpan model (opsional)
    # from joblib import dump
    # dump(model, 'sentiment_nb_smote_model.joblib')
    
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