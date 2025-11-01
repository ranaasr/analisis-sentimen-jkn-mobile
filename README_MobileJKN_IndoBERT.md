# MobileJKN Sentiment Analysis (IndoBERT/mBERT)

Notebook/pipeline ini membangun model **analisis sentimen biner (negatif/positif)** untuk ulasan aplikasi **MobileJKN** menggunakan **Hugging Face Transformers** (IndoBERT/mBERT). Bahasa pengantar: **Indonesia**.

---

## ✨ Fitur Utama
- **Cleaning & Labeling**: buang `NaN`, duplikat `reviewId`, komentar terlalu pendek; pemetaan skor → label biner (1–3=negatif, 4–5=positif).
- **Preprocess Bahasa Indonesia (Sastrawi)**: normalisasi slang → _tidak_; **negator dipertahankan**; stopword removal; stemming.
- **Model Transformers**: auto-pilih (`indolem/indobert-base-uncased` → `cahya/bert-base-indonesian-1.5G` → `bert-base-multilingual-cased`), **class weights**, scheduler **linear warmup**, **early stopping** berdasar **F1 validasi**.
- **Visualisasi**: kurva Loss/F1/Acc/AUC; **Confusion Matrix** tema **Blues**.
- **Error Analysis**: identifikasi **FP/FN** + ekspor `error_analysis.csv`.
- **Penyimpanan Artefak**: format Hugging Face (`save_pretrained`), `checkpoint.pt`, `train_history.csv`, `meta.json`.
- **Inference**: memuat model dari Drive/Working dir, preprocess Sastrawi ulang, prediksi batch input pengguna.

> Catatan: Vocab subword (WordPiece) mengikuti model yang diload. Contoh run yang terlihat: **`Vocab size: 31923`** (IndoBERT).

---

## 🧱 Dependensi & Lingkungan
- Python ≥ 3.9, **GPU** sangat disarankan (CUDA).
- Paket: `pandas`, `numpy`, `scikit-learn`, `Sastrawi`, `tqdm`, `joblib`, `torch`, `transformers`, `matplotlib`.

Instal cepat (Colab/Kaggle):
```bash
pip -q install pandas numpy scikit-learn Sastrawi tqdm joblib transformers
# Torch biasanya sudah ada di Colab/Kaggle.
```

---

## 📁 Dataset
- Asumsi data mentah: `/kaggle/input/dataset-jkn/mobilejkn_reviews_raw_id20000.csv` dengan kolom minimal:
  - `reviewId`, `timestamp`, `score`, `content`
- Data bersih (biner) akan disimpan ke (default contoh notebook):
  - `/kaggle/working/mobilejkn_reviews_clean_binary new.csv`

> **Penting**: Folder `/kaggle/input` adalah **read-only**. Jika ingin membaca ulang dataset bersih, baca dari **`/kaggle/working/`** (lihat bagian Troubleshooting).

---

## 🔀 Alur (Pipeline)
1. **Ingest & Inspect** — baca CSV mentah, cek kolom & rentang tanggal.
2. **Cleaning Dasar** — drop `NaN(content)`, drop duplikat `reviewId`, filter panjang konten > 2.
3. **Label Mapping (Biner)** — 1–3 → 0 (negatif), 4–5 → 1 (positif).
4. **Preprocess ID** — normalisasi slang, pertahankan **negator**, stopword removal, stemming (Sastrawi) → kolom `text`.
5. **Simpan CSV Bersih** — (biner) ke `/kaggle/working/...`.
6. **Split** — stratified train/val/test.
7. **Tokenisasi Model** — `AutoTokenizer` (subword WordPiece/BPE), `DataCollatorWithPadding`.
8. **Training** — CE Loss + **class weights**, Adam, linear warmup, **early stopping F1**.
9. **Plotting** — kurva Loss/F1/Acc/AUC + _best epoch_.
10. **Evaluasi** — metrik + **Confusion Matrix (Blues)**.
11. **Error Analysis** — `error_analysis.csv` (OK/FP/FN).
12. **Save Artefak** — `save_pretrained`, `checkpoint.pt`, `meta.json`.
13. **Inference** — load dari Drive/Working, preprocess lagi, prediksi batch input.

---

## 🧩 Struktur Notebook (ringkas)
- **Cell 1–2**: Install, versi, load CSV mentah + ringkasan.
- **Cell 3–4**: Cleaning & label mapping (biner).
- **Cell 5–6**: Preprocess (Sastrawi) + simpan dataset bersih.
- **Cell 7–8**: Split; (opsional) definisi TF-IDF word/char (tidak dipakai BERT).
- **Cell 2 (blok HF)**: Setup umum (seed/GPU).
- **Cell 3–10**: Load CSV bersih; dataset, tokenizer, collator, dataloader; model, class weights, optimizer, scheduler.
- **Cell 11–11A**: Train + **early stopping**, simpan `best`, plotting metrik.
- **Cell 12**: Evaluasi + **Confusion Matrix (Blues)**.
- **Cell 20**: Save artefak (model/tokenizer/checkpoint/history/meta).
- **Cell XX**: Error analysis (FP/FN) → `error_analysis.csv`.
- **Mount Drive**: (opsional, untuk Colab) `drive.mount(...)`.
- **Cell 21–22**: Load artefak & **inference** batch input (preprocess Sastrawi konsisten).

---

## 🚀 Cara Jalan Cepat (Kaggle/Colab)
1. **Pastikan path data mentah benar** di Cell 2.
2. Jalankan **Cell 1–6** untuk menghasilkan dataset bersih (biner).
3. Jalankan blok **HF (Cell 2–12)** untuk training + evaluasi.
4. (Opsional) Jalankan **Cell 20** untuk menyimpan artefak.
5. (Colab) **Copy** folder model ke Drive, lalu jalankan **Cell 21–22** untuk inference.

---

## ⚙️ Hyperparameter Penting
- `max_length=256` (tokenizer) — bisa dinaikkan ke 320/384/512 jika banyak input panjang dan VRAM cukup.
- **LR**: default contoh `1e-6` (konservatif). Disarankan coba `1e-5` atau `2e-5`.
- **Batch Size**: `8` (naikkan jika VRAM cukup).
- **Warmup**: `3%` langkah training (bisa ke `10%`).
- **Early Stopping**: `patience=2`, monitor `val_f1`.
- **Class Weights**: otomatis dari distribusi `y_train`.

---

## 📊 Evaluasi & Visualisasi
- Metrik: **Acc, Precision, Recall, F1 (binary), ROC-AUC**.
- **Confusion Matrix (Blues)** dengan anotasi angka.
- **Kurva**: Loss/F1/Acc (Train vs Val) + AUC (Val) dan garis vertikal **best epoch**.

---

## 🔎 Error Analysis
- Keluaran: `error_analysis.csv` berisi kolom `text`, `true_label`, `pred_label`, (opsional) `proba_pos/neg`.
- **Diagnosa cepat**: banyakkah **FP** pada kalimat bernada sarkas? Atau **FN** pada kalimat bernegator (“tidak”)? Pertimbangkan tuning threshold atau aturan kecil.

> **Jika hanya `proba_pos` yang tersedia**, set `proba_neg = 1 - proba_pos`.

---

## 🧪 Inference (contoh)
- **Load** model/tokenizer dari folder artefak (Drive/Working).  
- Jalankan **Cell 22**: masukkan beberapa kalimat (multi-baris), ENTER kosong untuk selesai.  
- Output: teks asli, hasil preprocess, `pred_label`, probabilitas.

Contoh cepat di kode:
```python
SAMPLE_TEXTS = [
    "Aplikasinya sering error pas login",
    "Pelayanan cepat, terima kasih BPJS"
]
# Ganti blok input() dengan SAMPLE_TEXTS jika perlu.
```

---

## 🧰 (Opsional) Baseline TF‑IDF
Vectorizer sudah didefinisikan (word 1–3gram, char 3–5gram) namun **tidak dipakai**.  
Jika ingin tambahkan baseline:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

pipe = make_pipeline(tfidf_word, LogisticRegression(max_iter=2000, n_jobs=-1))
pipe.fit(X_train, y_train)
y_hat = pipe.predict(X_test)
print(classification_report(y_test, y_hat, target_names=["negatif","positif"]))
```

---

## 🧯 Troubleshooting (umum)
1) **Path dataset bersih tidak konsisten**  
   - Simpan: `/kaggle/working/mobilejkn_reviews_clean_binary new.csv`  
   - Baca lagi dari **`/kaggle/working/...`** (bukan `/kaggle/input/...`).

2) **Probabilitas di Error Analysis kosong**  
   - `eval_and_report_loader` mengembalikan `probs_pos` (1-D). Isi:
   ```python
   err_df["proba_pos"] = test_probs
   err_df["proba_neg"] = 1 - test_probs
   ```
   - Atau ubah fungsi untuk mengembalikan `softmax(logits)` (n,2).

3) **AUC error (label 1 kelas saja)**  
   ```python
   auc = roc_auc_score(y_true, probs_pos) if len(np.unique(y_true))==2 else np.nan
   ```

4) **Training terlalu lambat / macet**  
   - Naikkan LR (`1e-5`/`2e-5`),/atau freeze sebagian layer awal BERT, gunakan batch lebih besar jika memungkinkan.

5) **OOM (kehabisan VRAM)**  
   - Turunkan `batch_size`, `max_length`, atau aktifkan gradient checkpointing (lanjutan).

---

## 📐 Reproducibility
- `set_seed(42)` untuk Python/NumPy/Torch (CPU/GPU).  
- Catat `meta.json` + `train_history.csv` untuk jejak eksperimen.

---

## 📜 Lisensi & Kredit
- Model dasar: **Hugging Face** (IndoBERT/mBERT). Patuhi lisensi masing‑masing repo.
- Kode notebook ini dapat dipakai untuk kebutuhan akademik. Tambahkan kredit sitasi bila dipublikasikan.

---

## 🙋 FAQ Singkat
- **Apakah perlu API token?** Tidak, kecuali mengakses repo privat di Hugging Face.
- **Token apa yang dipakai model?** Subword (WordPiece/BPE); contoh vocab size yang tercetak: **31,923**.
- **Bisa 3 kelas (negatif/netral/positif)?** Bisa — ubah label mapping, set `num_labels=3`, retrain.

---

## 🗺️ Roadmap (opsional)
- Tuning threshold via Precision‑Recall curve.
- StratifiedGroupKFold per pengguna (jika ada `userId`).
- Penambahan baseline klasik & ensemble ringan.
