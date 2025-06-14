Install Library : Install library yang diperlukan untuk menjalankan projek ini dengan cara membuka path di terminal dan ketik “python install -r requirements.txt” lalu enter
Run all preprocessing_dataset.ipynb : Kode ini bertujuan untuk melakukan pembersihan dan pra-pemrosesan pada ketiga dataset dan menghasilkan dua dataset baru, yakni dataset sms Bahasa Indonesia dan dataset sms Bahasa Inggris
Run all undersampling_oversampling.ipynb : Kode ini bertujuan untuk melakukan undersampling dan oversampling pada kedua dataset sebelumnya. Output dari kode ini adalah dua dataset baru, yakni dataset sms undersampling dan dataset sms oversampling
Run NaiveBayesManual; .py dan SVMManual.py : Kode ini berisi model Naive Bayes dan SVM secara manual yang nantinya akan dipanggil di mymain.ipynb.
Run all mymain.ipynbKode ini bertujuan untuk melakukan feature extraction/engineering TF-IDF pada kedua dataset, melatih model, dan mengevaluasi model, lalu menyimpannya menjadi file pickle
Run streamlit di dalam web.py : Kode ini bertujuan untuk menampilkan web streamlit dengan cara buka path terminal dan ketik “streamlit run web.py” lalu enter.
