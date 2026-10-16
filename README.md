# Latihan Machine Learning

Latihan ini membandingkan beberapa algoritma klasifikasi untuk memprediksi jurusan dan menyediakan aplikasi Streamlit untuk rekomendasi jurusan berdasarkan nilai dan peminatan.

## Isi proyek latihan

- `comparation.py` — melatih dan membandingkan Decision Tree, Random Forest, KNN, Naive Bayes, dan SVM berdasarkan akurasi serta waktu pelatihan.
- `app.py` — aplikasi Streamlit untuk merekomendasikan jurusan.
- `jurusan.csv` — dataset yang digunakan oleh kedua skrip.

## Persyaratan

- Python 3.11 (versi yang digunakan untuk proyek ini).

## Setup proyek

Buka PowerShell di folder proyek, lalu buat dan aktifkan virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Jika PowerShell menolak aktivasi skrip, jalankan perintah berikut satu kali pada akun Windows Anda, kemudian aktifkan kembali environment:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Pasang library yang dibutuhkan:

```powershell
pip install pandas scikit-learn streamlit
```

Jalankan perintah berikut dari folder utama proyek agar skrip dapat menemukan file dataset.

## Membandingkan algoritma

```powershell
python comparation.py
```

Hasil perbandingan akurasi dan waktu pelatihan setiap algoritma, serta algoritma dengan akurasi tertinggi, akan ditampilkan di terminal.

## Menjalankan aplikasi rekomendasi

Skrip aplikasi yang tersedia di proyek ini bernama `app.py`. Jalankan dengan:

```powershell
streamlit run app.py
```

Streamlit akan membuka aplikasi di browser. Pilih bulan penjualan, lalu klik **Rekomendasikan** untuk melihat hasilnya.

> **Catatan:** `app2.py` belum tersedia di proyek ini. Jika file tersebut sudah ditambahkan, perintah yang diminta dapat dijalankan dengan `streamlit run app2.py`. Untuk file yang ada saat ini, gunakan `streamlit run app.py`.
