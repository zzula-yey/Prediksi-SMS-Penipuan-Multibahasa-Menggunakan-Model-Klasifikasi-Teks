import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

model_fraud = pickle.load(open('file_pickle/model_fraud_undersampling.sav','rb'))
tfidf = TfidfVectorizer
loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle. load(open("./file_pickle/feature_tf-idf_undersampling.sav", "rb"))))

st.title ('Prediksi SMS Penipuan')
clean_teks = st.text_input('Masukan Teks SMS')
fraud_detection = ''
if st.button('Hasil Deteksi'):
    predict_fraud = model_fraud.predict(loaded_vec.fit_transform([clean_teks]).toarray())
    fraud_detection = 'SMS Normal' if predict_fraud[0] == 0 else 'SMS Penipuan'

st.success(fraud_detection)