import pandas as pd
from googletrans import Translator
from tqdm import tqdm

# Baca file CSV
file_path = "./data/kurikulum_OBE_Dataset(4).csv"  # Ganti dengan nama file CSV asli
output_path = "result.csv"

df = pd.read_csv(file_path)
translator = Translator()

# Terjemahkan kolom 'full_text'
def translate_text(text):
    try:
        return translator.translate(text, src='en', dest='id').text
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        return text  # Kembalikan teks asli jika terjadi kesalahan

# Tambahkan indikator progres
tqdm.pandas()
df['full_text'] = df['full_text'].progress_apply(translate_text)

# Simpan ke file baru
df.to_csv(output_path, index=False)
print(f"File hasil telah disimpan di {output_path}")
