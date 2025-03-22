import os
import csv
import random
import time
import pandas as pd
from tqdm import tqdm
from g4f.client import Client

# Konfigurasi
INPUT_CSV = './clean_data/processed_text_manual_annotation.csv'
PROGRESS_CSV = './labeled_data/progress_sentiment_manual_annotation.csv'
OUTPUT_CSV = './labeled_data/labeled_sentiment_gpt_for_cohens_kappa.csv'
BATCH_SIZE = 20
MAX_RETRIES = 5
DELAY_BETWEEN_RETRIES = 10

# Inisialisasi client g4f
client = Client()

# Template Prompt
PROMPT_TEMPLATE = """
Klasifikasikan teks berikut berdasarkan sentimen:
- positif
- negatif
- netral

Format respons: Nama kategori dipisahkan koma sesuai urutan teks. Hanya gunakan nama kategori resmi!

Contoh respons yang benar:
positif,netral,negatif,positif,netral

saya akan memberikan 20 record per setiap batch, maka pastikan response dari labelnya juga 20

Teks:
{}

Jawaban:"""

def load_data():
    try:
        if os.path.exists(PROGRESS_CSV):
            df_progress = pd.read_csv(PROGRESS_CSV)
            remaining = df_progress[df_progress['label'].isna()]
            print(f"🔄 Memuat progres sebelumnya. Sisa teks: {len(remaining)}")
            return remaining.to_dict('records')
        
        df = pd.read_csv(INPUT_CSV)
        df['label'] = None
        print(f"✅ Memuat dataset baru. Total teks: {len(df)}")
        return df.to_dict('records')
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        exit()

def create_batches(data):
    unlabeled = [d for d in data if pd.isna(d['label'])]
    random.shuffle(unlabeled)
    return [unlabeled[i:i+BATCH_SIZE] for i in range(0, len(unlabeled), BATCH_SIZE)]

def generate_prompt(batch):
    texts = [f"{idx+1}. {item['full_text']}" for idx, item in enumerate(batch)]
    return PROMPT_TEMPLATE.format('\n'.join(texts))

def parse_response(response, batch):
    valid_labels = ['positif', 'negatif', 'netral']
    
    try:
        raw_labels = [x.strip().lower() for x in response.split(',')]
        
        if len(raw_labels) != len(batch):
            raise ValueError(f"Jumlah label ({len(raw_labels)}) tidak sesuai batch ({len(batch)})")
            
        for i, label in enumerate(raw_labels):
            if label not in valid_labels:
                raise ValueError(f"Label tidak valid: '{label}'")
            batch[i]['label'] = label
        
        return batch
    except Exception as e:
        print(f"❌ Parsing error: {str(e)}")
        print(f"💬 Respons invalid: {response}")
        return None

def save_progress(data):
    df = pd.DataFrame(data)
    df.to_csv(PROGRESS_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"💾 Progress tersimpan: {len(df) - df['label'].isna().sum()}/{len(df)}")

def label_batch(batch):
    for attempt in range(MAX_RETRIES):
        try:
            prompt = generate_prompt(batch)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            result = parse_response(response.choices[0].message.content, batch)
            
            if result:
                labels = [item['label'] for item in result]
                label_counts = pd.Series(labels).value_counts().to_dict()
                print("🏷️ Label terdeteksi:")
                for lbl, count in label_counts.items():
                    print(f"  - {count}x {lbl}")
                
                return result
            return None
            
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1}/{MAX_RETRIES} gagal: {str(e)}")
            time.sleep(DELAY_BETWEEN_RETRIES)
    return None

def main():
    data = load_data()
    total_unlabeled = len([d for d in data if pd.isna(d['label'])])
    
    with tqdm(total=total_unlabeled, desc="🚀 Progress Labeling", unit="teks") as pbar:
        batches = create_batches(data)
        
        for batch in batches:
            print(f"\n🔖 Memproses {len(batch)} teks...")
            labeled_batch = label_batch(batch)
            
            if labeled_batch:
                for item in labeled_batch:
                    for original in data:
                        if original['full_text'] == item['full_text']:
                            original['label'] = item['label']
                            break
                
                save_progress(data)
                pbar.update(len(batch))
                
            else:
                print(f"❌ Batch gagal diproses setelah {MAX_RETRIES} percobaan")

    pd.DataFrame(data)[['full_text', 'tokens', 'processed_text', 'label']].to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"\n🎉 Labeling selesai! Hasil tersimpan di {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

            
            
            