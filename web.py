#######
import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model dan TF-IDF vectorizer
model_fraud = pickle.load(open('model_fraud_undersampling.sav', 'rb'))
loaded_vec = TfidfVectorizer(
    decode_error="replace",
    vocabulary=set(pickle.load(open("feature_tf-idf_undersampling.sav", "rb")))
)

# Panduan penggunaan
st.markdown("""
### 📘 Panduan Penggunaan
1. Masukkan teks SMS yang ingin diperiksa pada kolom input di bawah.
2. Klik tombol *"Hasil Deteksi"*.
3. Aplikasi akan menampilkan hasil prediksi apakah SMS tersebut *normal* atau *penipuan*.

💡 Tips:
- Gunakan teks SMS yang lengkap untuk hasil deteksi yang lebih akurat.
""")

# Judul aplikasi
st.title('📱 Prediksi SMS Penipuan')

# Input dari pengguna
clean_teks = st.text_input('✉ Masukkan Teks SMS')

# Inisialisasi hasil
fraud_detection = ''

# Tombol prediksi
if st.button('Hasil Deteksi'):
    predict_fraud = model_fraud.predict(loaded_vec.fit_transform([clean_teks]).toarray())
    fraud_detection = '✅ SMS Normal' if predict_fraud[0] == 0 else '⚠ SMS Penipuan'

# Tampilkan hasil
if fraud_detection:
    st.success(fraud_detection)
