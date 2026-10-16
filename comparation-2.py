import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# ==========================================
# Membaca Dataset
# ==========================================

df = pd.read_csv("dataset_penjualan_produk_bersih.csv")

print("="*60)
print("DATASET")
print("="*60)
print(df)

# ==========================================
# Encoding Data (kolom kategorikal -> angka)
# ==========================================

kolom_kategori = ["kategori", "ada_promo", "momen_hari_besar"]

for kolom in kolom_kategori:
    df[kolom] = LabelEncoder().fit_transform(df[kolom])

encoder_status = LabelEncoder()
df["status_penjualan"] = encoder_status.fit_transform(df["status_penjualan"])

# ==========================================
# Feature dan Target
# ==========================================
# Catatan: jumlah_terjual dan stok_awal_bulan TIDAK dipakai
# karena berkaitan langsung dengan target (data leakage).

X = df[["kategori",
        "bulan",
        "harga_satuan",
        "diskon_persen",
        "ada_promo",
        "biaya_iklan",
        "rating_produk",
        "momen_hari_besar"]]

y = df["status_penjualan"]

# ==========================================
# Membagi Data
# 70% Training
# 30% Testing
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y      # menjaga proporsi Laris/Biasa tetap sama
)

# ==========================================
# Daftar Algoritma
# KNN dan SVM memakai StandardScaler karena
# skala fitur (harga, biaya_iklan) sangat berbeda
# ==========================================

algorithms = {

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(random_state=42),

    "KNN":
        make_pipeline(StandardScaler(), KNeighborsClassifier()),

    "Naive Bayes":
        GaussianNB(),

    "SVM":
        make_pipeline(StandardScaler(), SVC())

}

hasil = []

print("\n")
print("="*60)
print("HASIL PERBANDINGAN")
print("="*60)

for nama, model in algorithms.items():

    start = time.time()

    model.fit(X_train, y_train)

    prediksi = model.predict(X_test)

    end = time.time()

    accuracy = accuracy_score(y_test, prediksi)

    waktu = end - start

    hasil.append([nama, accuracy, waktu])

    print(f"{nama:20s} Accuracy : {accuracy:.2%}   Training : {waktu:.5f} detik")

# ==========================================
# Menentukan Algoritma Terbaik
# ==========================================

terbaik = max(hasil, key=lambda x: x[1])

print("\n")
print("="*60)
print("ALGORITMA TERBAIK")
print("="*60)

print("Nama Algoritma :", terbaik[0])
print("Accuracy       :", f"{terbaik[1]:.2%}")
print("Training Time  :", f"{terbaik[2]:.5f} detik")
