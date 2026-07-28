import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# ==========================================
# Membaca Dataset
# ==========================================

df = pd.read_csv("jurusan.csv")

print("="*60)
print("DATASET")
print("="*60)
print(df)

# ==========================================
# Encoding Data
# ==========================================

encoder_minat = LabelEncoder()
df["minat"] = encoder_minat.fit_transform(df["minat"])

encoder_jurusan = LabelEncoder()
df["jurusan"] = encoder_jurusan.fit_transform(df["jurusan"])

# ==========================================
# Feature dan Target
# ==========================================

X = df[["matematika",
        "inggris",
        "pemrograman",
        "minat"]]

y = df["jurusan"]

# ==========================================
# Membagi Data
# 70% Training
# 30% Testing
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# ==========================================
# Daftar Algoritma
# ==========================================

algorithms = {

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(random_state=42),

    "KNN":
        KNeighborsClassifier(),

    "Naive Bayes":
        GaussianNB(),

    "SVM":
        SVC()

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