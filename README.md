# outcome-based-education-sentiment-analysis

Proyek ini bertujuan untuk melakukan analisis sentimen terhadap diskusi mengenai kurikulum Outcome-Based Education (OBE) di Twitter. Proses ini melibatkan scraping data, pembersihan data, normalisasi fitur, anotasi, serta pelatihan dan evaluasi beberapa model machine learning.

## Struktur Proyek

### 1. Scraping Data

- `Scrapping_twitter.ipynb`
  - Mengambil data dari Twitter mengenai kurikulum OBE.
  - Menggunakan API Twitter atau metode scraping lainnya untuk mengumpulkan data mentah.

### 2. Pembersihan dan Preprocessing Data

- `cleaning_data.py`
  - Membersihkan data dari karakter tidak diinginkan.
  - Melakukan tokenisasi, lemma, stopword removal, dan stemming.
  - Menyimpan data yang sudah diproses dalam format yang sesuai.

### 3. Normalisasi Fitur

- `word_to_vec.py`
  - Mengonversi teks menjadi representasi vektor.
  - Menggunakan teknik Word2Vec untuk mendapatkan representasi numerik dari teks.

### 4. Anotasi Data

- `labeling.py`
  - Melakukan anotasi data menggunakan API GPT-4o-mini.
  - Menentukan label sentimen seperti positif, netral, atau negatif.

### 5. Pelatihan Model Machine Learning

- `svm.ipynb`
  - Melatih model Support Vector Machine (SVM) menggunakan dataset yang sudah dinormalisasi.
  - Mengevaluasi performa model menggunakan metrik seperti akurasi, precision, recall, F1-score, dan ROC-AUC.
- `naive_bayes.ipynb`
  - Melatih model Naive Bayes untuk klasifikasi sentimen.
  - Mengevaluasi performa model menggunakan metrik seperti akurasi, precision, recall, F1-score, dan ROC-AUC.
- `fine_tuned_indoBERT.ipynb`
  - Melakukan fine-tuning model IndoBERT untuk meningkatkan akurasi dalam memahami sentimen dalam bahasa Indonesia.
  - Mengevaluasi hasil pelatihan menggunakan dataset yang telah dianotasi.

### 6. Analisis dan Visualisasi Data

- `label_distribution_visualization.ipynb`
  - Memvisualisasikan distribusi label dalam dataset menggunakan grafik batang dan pie chart.
- `cohens_kappa.ipynb`
  - Menilai tingkat kesepakatan antara anotasi yang dihasilkan oleh GPT-4o-mini dengan anotasi manual menggunakan Cohen's Kappa.
- `word_cloud.ipynb`
  - Membuat word cloud untuk menampilkan kata-kata yang paling sering muncul dalam setiap kategori sentimen.
  - Menyediakan heatmap untuk memahami hubungan antara berbagai topik dengan sentimen positif, netral, dan negatif.
- `visualization_comparison_model_performence.ipynb`
  - Membandingkan performa ketiga model (SVM, Naive Bayes, dan IndoBERT) melalui visualisasi grafik.
