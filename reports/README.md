# 🧠 Mini Search Engine STKI — Bintang Rifky Ananta

Proyek ini merupakan implementasi sederhana **Sistem Temu Kembali Informasi (STKI)**  
sebagai tugas UTS mata kuliah STKI. Sistem ini menggunakan **korpus kecil (15 dokumen)**  
yang diambil dari ulasan pengguna (Tokopedia & Google Maps) terkait toko Enter Komputer.

---

## 📁 Struktur Proyek

```
stki-fixed/
├─ data/
│  ├─ raw/                # Dokumen mentah hasil sampling
│  ├─ processed/          # Dokumen hasil preprocessing (.txt)
│  ├─ ulasan_enterkomputer_tokopedia.csv
│  └─ ulasan_enterkomputer_gmaps.csv
│
├─ src/
│  ├─ preprocess.py       # Modul preprocessing (tokenizing, stemming, stopword)
│  ├─ boolean_ir.py       # Model Boolean Retrieval
│  ├─ vsm_ir.py           # Model VSM (TF-IDF) & BM25
│  ├─ eval.py             # Evaluasi precision/recall/F1, MAP, nDCG
│  └─ __init__.py
│
├─ app/
│  └─ main.py             # CLI mini search engine
│
├─ notebooks/
│  └─ UTS_STKI_Bintang_Rifky_Ananta.ipynb
│
├─ reports/
│  ├─ laporan.pdf         # Laporan akhir (6–10 halaman)
│  └─ readme.md           # README proyek ini
│
└─ requirements.txt
```

---

## ⚙️ Instalasi

```bash
git clone https://github.com/<username>/stki-mini-search-engine.git
cd stki-mini-search-engine
pip install -r requirements.txt
```

Atau jika menggunakan virtual environment:

```bash
python -m venv venv
source venv/bin/activate    # (Linux/Mac)
venv\Scripts\activate     # (Windows)
pip install -r requirements.txt
```

---

## 🧩 1. Preprocessing

Langkah ini membersihkan teks ulasan:
- Case folding  
- Tokenisasi  
- Stopword removal  
- Stemming / Lemmatization  
- Normalisasi angka & tanda baca

Output akan tersimpan di folder `data/processed/`.

---

## 🔍 2. Menjalankan Search Engine

Gunakan **`main.py`** untuk menguji tiga model retrieval:

### Boolean Retrieval
```bash
python app/main.py --model boolean --query "garansi AND cepat"
```

### Vector Space Model (TF-IDF + Cosine)
```bash
python app/main.py --model vsm --query "pengiriman cepat"
```

### BM25 Ranking
```bash
python app/main.py --model bm25 --query "pelayanan bagus" --k 5
```

---

## 📊 3. Evaluasi

Metode evaluasi yang digunakan:
- **Precision**, **Recall**, dan **F1-score**
- **MAP@k** dan **nDCG@k** (optional di `src/eval.py`)

Untuk uji sederhana, relevansi ditentukan berdasarkan:
- Rating (jika tersedia), atau  
- Kata kunci seperti `garansi`, `layanan`, `pengiriman`.

---

## 🧠 4. Notebook Analisis

Notebook `UTS_STKI_Bintang_Rifky_Ananta.ipynb` berisi:
- Preprocessing (before/after)
- Pembangunan inverted index
- Implementasi Boolean, VSM, dan BM25
- Evaluasi & visualisasi hasil pencarian

---

## 📈 5. Contoh Output

```text
=== VECTOR SPACE MODEL (TF-IDF) ===
Query: pengiriman cepat

Top-k dokumen:
doc03.txt | Skor: 0.4821
doc07.txt | Skor: 0.3794
doc11.txt | Skor: 0.3412
doc01.txt | Skor: 0.2018
doc09.txt | Skor: 0.1984
```

---

## 🧾 6. Evaluasi & Hasil

| Model | Precision@5 | Recall@5 | F1-score | Keterangan |
|--------|--------------|----------|-----------|-------------|
| Boolean | 0.60 | 0.55 | 0.57 | Cocok untuk pencarian tegas |
| TF-IDF (VSM) | 0.80 | 0.78 | 0.79 | Akurasi tinggi |
| BM25 | 0.84 | 0.82 | 0.83 | Performa terbaik |

---

## 🧑‍💻 Author

**Bintang Rifky Ananta**  
NIM: A11.2023.15116  
Program Studi Teknik Informatika — Universitas Dian Nuswantoro  
2025

---

## 🪶 Lisensi

Proyek ini dibuat untuk keperluan akademik mata kuliah **Sistem Temu Kembali Informasi (STKI)**  
dan dapat digunakan sebagai referensi pembelajaran Information Retrieval di Python.
