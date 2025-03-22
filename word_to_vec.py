from gensim.models import Word2Vec
import pandas as pd
from tqdm import tqdm
import numpy as np

word2vec_output_path = "./preprocess_data/word2vec_vectors(withlabel).csv"  # File untuk hasil word2vec

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

df = pd.read_csv('./labeled_data/labeled_sentiment.csv')
tokens_list = df['tokens'].tolist()
vector_size = 100  # Dimensi vektor
tokens_list = [tokens for tokens in tokens_list if tokens]

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
doc_vectors_df['full_text'] = df['full_text']
doc_vectors_df['label'] = df['label']


# Simpan vektor ke file CSV
doc_vectors_df.to_csv(word2vec_output_path, index=False)
print(f"File hasil vektor Word2Vec telah disimpan di {word2vec_output_path}")