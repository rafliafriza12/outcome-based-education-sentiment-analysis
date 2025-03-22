import pandas as pd

# Baca file CSV
file_path = "./data/kurikulum_OBE_Dataset.csv"  # Ganti dengan nama file CSV asli
output_path = "result_dataset_obe.csv"

df = pd.read_csv(file_path)

# Pindahkan nilai dari 'translated_text' ke 'full_text'
df = df[['full_text']]

df.to_csv(output_path, index=False)
print(f"File hasil telah disimpan di {output_path}")