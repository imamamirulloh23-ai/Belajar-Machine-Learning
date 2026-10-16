import streamlit as st
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================
# Pengaturan Halaman
# ==========================
st.set_page_config(page_title="AI Prediksi Produk Laris")

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("dataset_penjualan_produk_bersih.csv")

# ==========================
# Encoding Kolom Kategorikal
# ==========================
kolom_kategori = ["kategori", "ada_promo", "momen_hari_besar"]

encoders = {}
for kolom in kolom_kategori:
    encoders[kolom] = LabelEncoder()
    df[kolom] = encoders[kolom].fit_transform(df[kolom])

# ==========================
# Feature & Target
# (jumlah_terjual dan stok_awal_bulan tidak dipakai
#  karena berkaitan langsung dengan target)
# ==========================
fitur = ["kategori", "bulan", "harga_satuan", "diskon_persen",
         "ada_promo", "biaya_iklan", "rating_produk", "momen_hari_besar"]

X = df[fitur]
y = df["status_penjualan"]

# ==========================
# Training Model
# ==========================
@st.cache_resource
def latih_model():
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model

model = latih_model()

# ==========================
# Tampilan
# ==========================
st.title("🛒 AI Prediksi Produk Laris")

st.write("Masukkan kondisi produk dan bulan penjualan.")

nama_bulan = ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
              "Juli", "Agustus", "September", "Oktober", "November", "Desember"]

kategori = st.selectbox("Kategori Produk", encoders["kategori"].classes_)

bulan_nama = st.selectbox("Bulan Penjualan", nama_bulan)

momen = st.selectbox("Momen Hari Besar", encoders["momen_hari_besar"].classes_)

harga = st.number_input("Harga Satuan (Rp)", 1000, 1000000, 100000, step=1000)

diskon = st.slider("Diskon (%)", 0, 30, 10)

promo = st.selectbox("Ada Promo", encoders["ada_promo"].classes_)

iklan = st.number_input("Biaya Iklan (Rp)", 0, 9000000, 3000000, step=100000)

rating = st.slider("Rating Produk", 3.0, 5.0, 4.3, step=0.1)

# Encode input
bulan = nama_bulan.index(bulan_nama) + 1
kategori_encode = encoders["kategori"].transform([kategori])[0]
promo_encode = encoders["ada_promo"].transform([promo])[0]
momen_encode = encoders["momen_hari_besar"].transform([momen])[0]

# ==========================
# Prediksi
# ==========================
if st.button("Prediksi"):

    data_input = pd.DataFrame(
        [[kategori_encode, bulan, harga, diskon,
          promo_encode, iklan, rating, momen_encode]],
        columns=fitur
    )

    hasil = model.predict(data_input)[0]
    peluang = model.predict_proba(data_input)[0]
    peluang_laris = peluang[list(model.classes_).index("Laris")]

    if hasil == "Laris":
        st.success(f"Prediksi : **{hasil}** 🎉")
    else:
        st.info(f"Prediksi : **{hasil}**")

    st.write(f"Peluang Laris : **{peluang_laris:.0%}**")
