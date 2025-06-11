import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('stopwords')


class TextPreprocessor:
    def __init__(self, key_norm_dataset=None, language='indonesian'):
        self.language = language
        self.set_stopwords(language)
        self.set_stemmer(language)
        self.key_norm = key_norm_dataset

    def set_stopwords(self, language):
        try:
            self.stopwords_list = stopwords.words(language)
        except:
            print(f"[PERINGATAN] Stopwords untuk bahasa '{language}' tidak ditemukan. Stopword akan dikosongkan.")
            self.stopwords_list = []

    def set_stemmer(self, language):
        if language == 'indonesian':
            self.stemmer = StemmerFactory().create_stemmer()
        elif language == 'english':
            self.stemmer = PorterStemmer()
        else:
            self.stemmer = None

    def preprocess_first(self, teks):
        teks = teks.lower()
        teks = re.sub(r'[-+]?[0-9]+', '', teks)
        teks = re.sub(r'https?://\S+|www\.\S+', '', teks)
        teks = re.sub(r'[^\w\s]', '', teks)
        teks = teks.strip()
        return teks

    def text_normalize(self, teks):
        if self.key_norm is not None:
            teks = ' '.join([
                self.key_norm[self.key_norm['singkat'] == word]['hasil'].values[0]
                if (self.key_norm['singkat'] == word).any()
                else word for word in teks.split()
            ])
        return teks.lower()

    def remove_stop_words(self, teks):
        words = teks.split()
        clean_words = [word for word in words if word not in self.stopwords_list]
        return ' '.join(clean_words)

    def stemming(self, teks):
        if isinstance(self.stemmer, PorterStemmer):
            return ' '.join([self.stemmer.stem(word) for word in teks.split()])
        elif self.language == 'indonesian' and self.stemmer is not None:
            return self.stemmer.stem(teks)
        return teks  # jika stemmer tidak tersedia

    def preprocess(self, teks):
        if isinstance(teks, (int, float)):
            teks = str(teks)
        teks = self.preprocess_first(teks)
        teks = self.text_normalize(teks)
        teks = self.remove_stop_words(teks)
        teks = self.stemming(teks)
        return teks
