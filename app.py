import streamlit as st
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("jurusan.csv")

# ==========================
# Encoding Kolom Minat
# ==========================
encoder = LabelEncoder()
df["minat"] = encoder.fit_transform(df["minat"])

# ==========================
# Feature & Target
# ==========================
X = df[["matematika","inggris","pemrograman","minat"]]
y = df["jurusan"]

# ==========================
# Training Model
# ==========================
model = DecisionTreeClassifier(random_state=42)

model.fit(X,y)

# ==========================
# Tampilan
# ==========================
st.set_page_config(page_title="AI Rekomendasi Jurusan")

st.title("🎓 AI Rekomendasi Jurusan Kuliah")

st.write("Masukkan nilai siswa.")

mat = st.slider("Nilai Matematika",0,100,80)

eng = st.slider("Nilai Bahasa Inggris",0,100,80)

prog = st.slider("Nilai Pemrograman",0,100,80)

minat = st.selectbox(
    "Minat",
    ["IT","Bisnis","Desain"]
)

# Encode Minat
minat_encode = encoder.transform([minat])[0]

# ==========================
# Prediksi
# ==========================
if st.button("Prediksi"):

    hasil = model.predict([[mat,eng,prog,minat_encode]])

    st.success(f"Rekomendasi Jurusan : **{hasil[0]}**")