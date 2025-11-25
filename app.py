from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# =====================================================
# Load data & model
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Tingkat_Kemiskinan_Indonesia.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_kemiskinan.pkl")

df = pd.read_csv(DATA_PATH)

# --- bikin ringkasan untuk dashboard ---
total_provinsi = df['Provinsi'].nunique()
total_kabkota = df['Kab/Kota'].nunique()
total_miskin = (df['Klasifikasi Kemiskinan'] == 1).sum()
total_tidak_miskin = (df['Klasifikasi Kemiskinan'] == 0).sum()

avg_p0 = df['Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)'].mean()
avg_ipm = df['Indeks Pembangunan Manusia'].mean()
avg_pengeluaran = df['Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)'].mean()

# list provinsi untuk dropdown
provinsi_list = sorted(df['Provinsi'].unique())

# --- load model ---
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# fitur yang dipakai (HARUS sama dengan notebook)
FITUR = [
    'Provinsi_lbl',
    'KabKota_lbl',
    'Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)',
    'Rata-rata Lama Sekolah Penduduk 15+ (Tahun)',
    'Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)',
    'Indeks Pembangunan Manusia',
    'Umur Harapan Hidup (Tahun)',
    'Persentase rumah tangga yang memiliki akses terhadap sanitasi layak',
    'Persentase rumah tangga yang memiliki akses terhadap air minum layak',
    'Tingkat Pengangguran Terbuka',
    'Tingkat Partisipasi Angkatan Kerja',
    'PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)',
]

# untuk encoding provinsi/kabkota (sama logika dengan notebook)
from sklearn.preprocessing import LabelEncoder
label_prov = LabelEncoder()
label_kab = LabelEncoder()
df['Provinsi_lbl'] = label_prov.fit_transform(df['Provinsi'])
df['KabKota_lbl'] = label_kab.fit_transform(df['Kab/Kota'])


# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def dashboard():
    """
    Dashboard utama:
    - ringkasan nasional
    - filter provinsi (optional)
    - tabel data (untuk DataTables)
    """
    selected_prov = request.args.get("provinsi", "ALL")

    if selected_prov == "ALL":
        df_view = df.copy()
        judul = "Semua Provinsi di Indonesia"
    else:
        df_view = df[df['Provinsi'] == selected_prov].copy()
        judul = f"Provinsi {selected_prov}"

    # data ringkasan untuk provinsi terpilih
    prov_total_kabkota = df_view['Kab/Kota'].nunique()
    prov_miskin = (df_view['Klasifikasi Kemiskinan'] == 1).sum()
    prov_tidak_miskin = (df_view['Klasifikasi Kemiskinan'] == 0).sum()
    prov_avg_p0 = df_view['Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)'].mean()
    prov_avg_ipm = df_view['Indeks Pembangunan Manusia'].mean()

    # pilih kolom yang ditampilkan di tabel
    kolom_tabel = [
        'Provinsi',
        'Kab/Kota',
        'Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)',
        'Indeks Pembangunan Manusia',
        'Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)',
        'Umur Harapan Hidup (Tahun)',
        'Klasifikasi Kemiskinan'
    ]
    df_tabel = df_view[kolom_tabel].copy()

    # kirim ke template dalam bentuk list of dict
    data_tabel = df_tabel.to_dict(orient="records")

    return render_template(
        "dashboard.html",
        judul=judul,
        provinsi_list=provinsi_list,
        selected_prov=selected_prov,
        # ringkasan nasional
        total_provinsi=total_provinsi,
        total_kabkota=total_kabkota,
        total_miskin=int(total_miskin),
        total_tidak_miskin=int(total_tidak_miskin),
        avg_p0=avg_p0,
        avg_ipm=avg_ipm,
        avg_pengeluaran=avg_pengeluaran,
        # ringkasan prov
        prov_total_kabkota=prov_total_kabkota,
        prov_miskin=int(prov_miskin),
        prov_tidak_miskin=int(prov_tidak_miskin),
        prov_avg_p0=prov_avg_p0,
        prov_avg_ipm=prov_avg_ipm,
        data_tabel=data_tabel
    )


@app.route("/model-info")
def model_info():
    """
    Halaman penjelasan model + grafik evaluasi.
    """
    return render_template("model_info.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Halaman prediksi:
    - user pilih provinsi
    - user pilih kab/kota (filtered di frontend)
    - backend ambil baris data dari CSV & prediksi
    """
    hasil = None
    detail_row = None
    prob_miskin = None
    prob_tidak_miskin = None

    # mapping provinsi -> list kab/kota
    mapping = (
        df.groupby("Provinsi")["Kab/Kota"]
        .apply(list)
        .to_dict()
    )

    if request.method == "POST":
        prov = request.form.get("provinsi")
        kab = request.form.get("kabkota")

        row = df[(df['Provinsi'] == prov) & (df['Kab/Kota'] == kab)]
        if row.empty:
            hasil = "Data untuk kombinasi Provinsi & Kab/Kota tidak ditemukan."
        else:
            row = row.iloc[0]

            # buat baris fitur sesuai urutan
            prov_lbl = label_prov.transform([row['Provinsi']])[0]
            kab_lbl = label_kab.transform([row['Kab/Kota']])[0]

            x_input = [
                prov_lbl,
                kab_lbl,
                row['Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)'],
                row['Rata-rata Lama Sekolah Penduduk 15+ (Tahun)'],
                row['Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)'],
                row['Indeks Pembangunan Manusia'],
                row['Umur Harapan Hidup (Tahun)'],
                row['Persentase rumah tangga yang memiliki akses terhadap sanitasi layak'],
                row['Persentase rumah tangga yang memiliki akses terhadap air minum layak'],
                row['Tingkat Pengangguran Terbuka'],
                row['Tingkat Partisipasi Angkatan Kerja'],
                row['PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)'],
            ]

            import numpy as np
            x_input = np.array(x_input).reshape(1, -1)

            pred = model.predict(x_input)[0]
            prob = model.predict_proba(x_input)[0]
            prob_tidak_miskin = prob[0]
            prob_miskin = prob[1]

            hasil = "MISKIN (label 1)" if pred == 1 else "TIDAK MISKIN (label 0)"

            detail_row = {
                "Provinsi": row['Provinsi'],
                "Kab/Kota": row['Kab/Kota'],
                "P0": row['Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)'],
                "IPM": row['Indeks Pembangunan Manusia'],
                "Pengeluaran": row['Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)'],
                "UmurHarapan": row['Umur Harapan Hidup (Tahun)'],
                "Sanitasi": row['Persentase rumah tangga yang memiliki akses terhadap sanitasi layak'],
                "AirMinum": row['Persentase rumah tangga yang memiliki akses terhadap air minum layak'],
                "TPT": row['Tingkat Pengangguran Terbuka'],
                "TPAK": row['Tingkat Partisipasi Angkatan Kerja'],
                "PDRB": row['PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)'],
                "LabelAsli": row['Klasifikasi Kemiskinan']
            }

    return render_template(
        "predict.html",
        provinsi_list=provinsi_list,
        mapping_kab=mapping,
        hasil=hasil,
        detail_row=detail_row,
        prob_miskin=prob_miskin,
        prob_tidak_miskin=prob_tidak_miskin
    )


if __name__ == "__main__":
    app.run(debug=True)
