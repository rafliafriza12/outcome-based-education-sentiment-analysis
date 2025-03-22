import pandas as pd
import re
import string
import emoji
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import WordNetLemmatizer
import nltk
from tqdm import tqdm
import numpy as np

# Unduh resource NLTK yang diperlukan
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Open Multilingual WordNet

# inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Path file input dan output
file_path = "./data/dataset_manual_annotation.csv"  # Sesuaikan dengan file CSV asli
processed_output_path = "./clean_data/processed_text_manual_annotation.csv"  # File untuk processed text
word2vec_output_path = "./preprocess_data/word2vec_vectors_manual_annotation.csv"  # File untuk hasil word2vec

try:
    # df = pd.read_csv(file_path, sep=',', engine="python", encoding="utf-8", on_bad_lines="skip")
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error membaca file CSV: {e}")
    exit()

# Tampilkan beberapa baris pertama untuk pengecekan
print("Preview data:")
print(df.head())

# Pilih hanya kolom 'full_text'
df = df[['full_text']].astype(str)

# Kamus kata tidak baku ke kata baku
kamus_tidak_baku = {
    'gt': 'begitu',
    'gw': 'saya',
    'dgn': 'dengan',
    'utk': 'untuk',
    'yg': 'yang',
    'spy': 'supaya',
    'skrg': 'sekarang',
    'blm': 'belum',
    'sdh': 'sudah',
    'pd': 'pada',
    'dr': 'dari',
    'krn': 'karena',
    'bs': 'bisa',
    'tdk': 'tidak',
    'jd': 'jadi',
    'sm': 'sama',
    'jg': 'juga',
    'aja': 'saja',
    'bgmn': 'bagaimana',
    'byk': 'banyak',
    'sblm': 'sebelum',
    'stlh': 'setelah',
    'spt': 'seperti',
    'msh': 'masih',
    'hrs': 'harus',
    'sgt': 'sangat',
    'tp': 'tetapi',
    'klo': 'kalau',
    'gmn': 'bagaimana',
    'aja': 'saja',
    'udah': 'sudah',
    'gak': 'tidak',
    'ga': 'tidak',
    'nggak': 'tidak',
    'ngga': 'tidak',
    'gtu': 'begitu',
    'gini': 'begini',
    'bgt': 'banget',
    'banget': 'sekali',
    'kyk': 'seperti',
    'kayak': 'seperti',
    'sih': '',
    'deh': '',
    'dong': '',
    'loh': '',
    'lho': '',
    'kek': 'seperti',
    'cuman': 'hanya',
    'cuma': 'hanya',
    'doang': 'saja',
    'emang': 'memang',
    'gue': 'saya',
    'elu': 'kamu',
    'lo': 'kamu',
    'lu': 'kamu',
    'km': 'kamu',
    'sy': 'saya',
    'aq': 'saya',
    'aku': 'saya',
    'ak': 'saya',
    'dpt': 'dapat',
    'pny': 'punya',
    'mau': 'ingin',
    'dlm': 'dalam',
    'nnti': 'nanti',
    'ntar': 'nanti',
    'bru': 'baru',
    'sngt': 'sangat',
    'slalu': 'selalu',
    'kpn': 'kapan',
    'jgn': 'jangan',
    'blh': 'boleh',
    'pke': 'pakai',
    'pake': 'pakai',
    'sbg': 'sebagai',
    'mngkn': 'mungkin',
    'mgkn': 'mungkin',
    'dri': 'dari',
    'sma': 'sama',
    'tuh': 'itu',
    'wkt': 'waktu',
    'knp': 'kenapa',
    'mslh': 'masalah',
    'mslnya': 'misalnya',
    'bkn': 'bukan',
    'krna': 'karena',
    'lwt': 'lewat',
    'cb': 'coba',
    'tmn': 'teman',
    'br': 'baru',
    'hr': 'hari',
    'bln': 'bulan',
    'thn': 'tahun',
    'sblmnya': 'sebelumnya',
    'stlhnya': 'setelahnya',
    'brp': 'berapa',
    'jml': 'jumlah',
    'cmn': 'hanya',
    'sbnrnya': 'sebenarnya',
    'bnyk': 'banyak',
    'nanya': 'bertanya',
    'smua': 'semua',
    'smpe': 'sampai',
    'sampe': 'sampai',
    'tmpt': 'tempat',
    'wlpn': 'walaupun',
    'mnjd': 'menjadi',
    'slh': 'salah',
    'slsai': 'selesai',
    'nyangka': 'menyangka',
    'msk': 'masuk',
    'ktmu': 'ketemu',
    'ktmuan': 'pertemuan',
    'wlw': 'walau',
    'dah': 'sudah',
    'dri': 'dari',
    'sndri': 'sendiri',
    'brbeda': 'berbeda',
    'brjalan': 'berjalan',
    'pgen': 'ingin',
    'pgn': 'ingin',
    'kgn': 'kangen',
    'kangen': 'rindu',
    'msuk': 'masuk',
    'pst': 'pasti',
    'ngmng': 'ngomong',
    'ngomong': 'berbicara',
    'ngomongin': 'membicarakan',
    'bwh': 'bawah',
    'ats': 'atas',
    'dpn': 'depan',
    'blkng': 'belakang',
    'blakang': 'belakang',
    'kayaknya': 'sepertinya',
    'kalo': 'kalau'
}

# Kamus lemmatization untuk Bahasa Indonesia
kamus_lemma_indonesia = {
    'berbasis': 'basis',
    'praktekkan': 'praktek',
    'memberikan': 'beri',
    'diberikan': 'beri',
    'memberi': 'beri',
    'diri': 'diri',
    'berbicara': 'bicara',
    'pembicaraan': 'bicara',
    'dibicarakan': 'bicara',
    'membicarakan': 'bicara',
    'pembelajaran': 'ajar',
    'belajar': 'ajar',
    'mengajar': 'ajar',
    'pelajaran': 'ajar',
    'mempelajari': 'ajar',
    'pengajaran': 'ajar',
    'mengatakan': 'kata',
    'berkata': 'kata',
    'dikatakan': 'kata',
    'perkataan': 'kata',
    'diketahui': 'tahu',
    'mengetahui': 'tahu',
    'pengetahuan': 'tahu',
    'ketahui': 'tahu',
    'menuliskan': 'tulis',
    'tulisan': 'tulis',
    'dituliskan': 'tulis',
    'menulis': 'tulis',
    'berlari': 'lari',
    'larian': 'lari',
    'berlarian': 'lari',
    'membaca': 'baca',
    'dibaca': 'baca',
    'bacaan': 'baca',
    'terbaca': 'baca',
    'melihat': 'lihat',
    'terlihat': 'lihat',
    'penglihatan': 'lihat',
    'dilihat': 'lihat',
    'menyimak': 'simak',
    'tersimak': 'simak',
    'disimak': 'simak',
    'penyimakan': 'simak',
    'mengingat': 'ingat',
    'diingat': 'ingat',
    'ingatan': 'ingat',
    'teringat': 'ingat',
    'mendengar': 'dengar',
    'didengar': 'dengar',
    'terdengar': 'dengar',
    'pendengaran': 'dengar',
    'berlangsung': 'langsung',
    'melangsung': 'langsung',
    'dilangsung': 'langsung',
    'pelaksanaan': 'laksana',
    'melaksanakan': 'laksana',
    'dilaksanakan': 'laksana',
    'terlaksana': 'laksana',
    'mengerjakan': 'kerja',
    'bekerja': 'kerja',
    'pekerjaan': 'kerja',
    'dikerjakan': 'kerja',
    'perasaan': 'rasa',
    'merasa': 'rasa',
    'dirasakan': 'rasa',
    'terasa': 'rasa',
    'merasakan': 'rasa',
    'keadaan': 'ada',
    'mengada': 'ada',
    'adanya': 'ada',
    'berada': 'ada'
}

# Fungsi untuk mengoreksi kata tidak baku
def koreksi_kata_tidak_baku(text):
    words = text.split()
    corrected_words = []
    
    for word in words:
        if word in kamus_tidak_baku:
            corrected = kamus_tidak_baku[word]
            if corrected:  # Jika bukan string kosong
                corrected_words.append(corrected)
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)

# Fungsi untuk lemmatization bahasa Indonesia
def lemmatize_indonesian(tokens):
    lemmatized_tokens = []
    
    for token in tokens:
        # Cek apakah ada di kamus lemmatization Indonesia
        if token in kamus_lemma_indonesia:
            lemmatized_tokens.append(kamus_lemma_indonesia[token])
        else:
            # Gunakan WordNet lemmatizer sebagai fallback
            lemmatized_tokens.append(stemmer.stem(token))
            
    return lemmatized_tokens

# Fungsi pembersihan teks
def clean_text(text):
    text = text.lower()  # Case folding
    text = re.sub(r'http\S+', '', text)  # Hapus URL
    text = emoji.replace_emoji(text, replace='')  # Hapus emoji
    text = text.translate(str.maketrans('', '', string.punctuation))  # Hapus tanda baca
    text = koreksi_kata_tidak_baku(text)  # Koreksi kata tidak baku
    return text

# Fungsi tokenisasi
def tokenize(text):
    try:
        return word_tokenize(text)
    except Exception as e:
        print(f"Error saat tokenisasi: {e}")
        return []

# Load stopwords bahasa Indonesia
try:
    stop_words = set(stopwords.words('indonesian'))
except Exception as e:
    print(f"Error saat memuat stopwords: {e}")
    stop_words = set()

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

# Fungsi untuk mendapatkan vektor Word2Vec untuk dokumen
def get_document_vector(doc_tokens, model, vector_size):
    vec = np.zeros(vector_size)
    count = 0
    for word in doc_tokens:
        try:
            if word in model.wv:
                vec += model.wv[word]
                count += 1
        except KeyError:
            continue
    
    if count != 0:
        vec /= count
    return vec

# Inisialisasi lemmatizer
lemmatizer = WordNetLemmatizer()

# Terapkan preprocessing dengan progress bar
tqdm.pandas(desc="Memproses teks")
df['full_text'] = df['full_text'].progress_apply(clean_text)
df['tokens'] = df['full_text'].progress_apply(tokenize)
df['tokens'] = df['tokens'].progress_apply(remove_stopwords)
df['tokens'] = df['tokens'].progress_apply(lemmatize_indonesian)

df['processed_text'] = df['tokens'].progress_apply(lambda x: ' '.join(x))

# Simpan hasil processed_text ke file CSV terpisah
processed_df = df[['full_text', 'tokens', 'processed_text']]
processed_df.to_csv(processed_output_path, index=False)
print(f"File hasil processed text telah disimpan di {processed_output_path}")

# Latih model Word2Vec
print("Melatih model Word2Vec...")
tokens_list = df['tokens'].tolist()
vector_size = 100  # Dimensi vektor

# Filter token list yang kosong
tokens_list = [tokens for tokens in tokens_list if tokens]

# Latih model Word2Vec
model = Word2Vec(sentences=tokens_list, vector_size=vector_size, window=5, min_count=1, workers=4)

# Simpan model
model_output_path = "word2vec_model.model"
model.save(model_output_path)
print(f"Model Word2Vec telah disimpan di {model_output_path}")

# Generate vektor dokumen
print("Menghasilkan vektor dokumen...")
document_vectors = []

for tokens in tqdm(df['tokens'], desc="Menghasilkan vektor dokumen"):
    vec = get_document_vector(tokens, model, vector_size)
    document_vectors.append(vec)

# Konversi vektor dokumen ke DataFrame
doc_vectors_df = pd.DataFrame(document_vectors)
doc_vectors_df.columns = [f"feature_{i}" for i in range(vector_size)]

# Simpan vektor ke file CSV
doc_vectors_df.to_csv(word2vec_output_path, index=False)
print(f"File hasil vektor Word2Vec telah disimpan di {word2vec_output_path}")

# Visualisasi fitur kata (opsional - memerlukan library tambahan)
print("Anda dapat menggunakan model Word2Vec untuk menganalisis:")
print("1. model.wv.most_similar('kata') - untuk menemukan kata yang paling mirip")
print("2. model.wv.similarity('kata1', 'kata2') - untuk menghitung kesamaan antara dua kata")

print("Proses selesai!")