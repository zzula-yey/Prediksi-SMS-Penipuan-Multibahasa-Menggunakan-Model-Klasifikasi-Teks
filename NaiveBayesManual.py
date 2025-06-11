import numpy as np
import math

class NaiveBayesSimple:
    def __init__(self):
        self.class_probs = {}       # P(class)
        self.feature_probs = {}     # P(feature | class)

    def train(self, X, y):
        n_samples, n_features = X.shape

        for c in [0, 1]:
            # Ambil semua data untuk kelas c
            X_c = X[y == c]

            # Hitung probabilitas kelas
            self.class_probs[c] = len(X_c) / n_samples

            # Hitung jumlah total semua kata dalam kelas c
            total_words = np.sum(X_c) + n_features  # +n_features = Laplace smoothing
            word_counts = np.sum(X_c, axis=0) + 1   # +1 = Laplace smoothing

            # Simpan probabilitas kata per kelas
            self.feature_probs[c] = word_counts / total_words

        return self

    def predict(self, X):
        predictions = []

        for x in X:
            class_scores = {}

            for c in [0, 1]:
                # Mulai dari log(P(class))
                log_prob = math.log(self.class_probs[c])

                # Tambahkan log(P(feature|class)^count) = count * log(P(feature|class))
                for i in range(len(x)):
                    prob = self.feature_probs[c][i]
                    log_prob += x[i] * math.log(prob)

                class_scores[c] = log_prob

            # Pilih kelas dengan skor tertinggi
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)

        return predictions
