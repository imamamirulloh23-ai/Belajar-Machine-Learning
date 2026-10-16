import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================
# Pengaturan Halaman
# ==========================
st.set_page_config(page_title="AI Rekomendasi Produk")

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("dataset_penjualan_produk_bersih.csv")

# Simpan salinan asli untuk ringkasan & tampilan
df_asli = df.copy()

# ==========================
# Encoding Kolom Kategorikal
# ==========================
encoder_produk = LabelEncoder()
df["nama_produk"] = encoder_produk.fit_transform(df["nama_produk"])

encoder_momen = LabelEncoder()
df["momen_hari_besar"] = encoder_momen.fit_transform(df["momen_hari_besar"])

# ==========================
# Feature & Target
# Karena pengguna hanya memasukkan bulan, fitur dibatasi pada
# produk, bulan, dan momen hari besar (momen ditentukan dari bulan).
# jumlah_terjual dan stok_awal_bulan tidak dipakai (data leakage).
# ==========================
fitur = ["nama_produk", "bulan", "momen_hari_besar"]

X = df[fitur]
y = df["status_penjualan"]

# ==========================
# Training Model
# ==========================
@st.cache_resource
def latih_model():
    model = RandomForestClassifier(n_estimators=300, max_features=None, random_state=42)
    model.fit(X, y)
    return model

model = latih_model()

nama_bulan = ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
              "Juli", "Agustus", "September", "Oktober", "November", "Desember"]

kategori_produk = df_asli.groupby("nama_produk")["kategori"].first()

# ==========================
# Tampilan
# ==========================
st.title("🛒 AI Rekomendasi Produk untuk Dijual")

st.write("Pilih bulan, lalu AI akan merekomendasikan produk yang berpeluang laris.")

bulan_nama = st.selectbox("Bulan Penjualan", nama_bulan)

bulan = nama_bulan.index(bulan_nama) + 1

# ==========================
# Prediksi
# ==========================
if st.button("Rekomendasikan"):

    # Momen hari besar yang paling umum pada bulan tersebut
    momen = df_asli[df_asli["bulan"] == bulan]["momen_hari_besar"].mode()[0]

    daftar_produk = list(encoder_produk.classes_)

    data_input = pd.DataFrame({
        "nama_produk": encoder_produk.transform(daftar_produk),
        "bulan": bulan,
        "momen_hari_besar": encoder_momen.transform([momen])[0],
    })[fitur]

    idx_laris = list(model.classes_).index("Laris")
    peluang = model.predict_proba(data_input)[:, idx_laris]

    # Rata-rata penjualan historis tiap produk pada bulan tersebut
    rata_jual = (df_asli[df_asli["bulan"] == bulan]
                 .groupby("nama_produk")["jumlah_terjual"].mean().round(0))

    hasil = pd.DataFrame({
        "Produk": daftar_produk,
        "Kategori": [kategori_produk[p] for p in daftar_produk],
        "peluang": peluang,
        "Rata-rata Terjual (unit)": [int(rata_jual[p]) for p in daftar_produk],
    })

    rekomendasi = hasil[hasil["peluang"] >= 0.5].sort_values("peluang", ascending=False)

    if rekomendasi.empty:
        rekomendasi = hasil.sort_values("Rata-rata Terjual (unit)", ascending=False).head(3)
        st.info(f"Bulan {bulan_nama} tidak ada produk dengan lonjakan penjualan khusus. "
                "Berikut 3 produk dengan rata-rata penjualan tertinggi.")
    else:
        st.success(f"Produk yang direkomendasikan untuk bulan {bulan_nama} "
                   f"(momen: {momen}) 🎉")

    rekomendasi["Peluang Laris"] = (rekomendasi["peluang"] * 100).round(0).astype(int).astype(str) + "%"

    st.dataframe(
        rekomendasi[["Produk", "Kategori", "Peluang Laris", "Rata-rata Terjual (unit)"]],
        hide_index=True,
        use_container_width=True
    )
