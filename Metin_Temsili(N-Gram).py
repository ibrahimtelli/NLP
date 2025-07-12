# Gerekli kütüphaneleri içe aktar
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "Doğal dil işleme, bilgisayarların insan dilini anlamasına ve işlemesine olanak tanır.",
    "N-gram modelleri, metinlerdeki kelime dizilerini analiz etmek için kullanılır.",
    "Makine öğrenmesi algoritmaları, büyük veri kümelerinden öğrenme yeteneğine sahiptir.",
    "Python, veri bilimi ve yapay zeka alanında popüler bir programlama dilidir.",
    "Bu çalışma, metin analizi ve doğal dil işleme konularına odaklanmıştır."
]

# Unigram, Bigram ve Trigram için CountVectorizer modellerini oluşturma

# Unigram modeli (tek kelimeler)
vectorizer_unigram = CountVectorizer(ngram_range=(1, 1), stop_words='english', lowercase=True)

# Bigram modeli (iki kelimelik diziler)
vectorizer_bigram = CountVectorizer(ngram_range=(2, 2), stop_words='english', lowercase=True)

# Trigram modeli (üç kelimelik diziler)
vectorizer_trigram = CountVectorizer(ngram_range=(3, 3), stop_words='english', lowercase=True)

# --- Unigram Analizi ---
# Metinleri unigram modeline dönüştür ve özellikleri çıkar
X_unigram = vectorizer_unigram.fit_transform(documents)
unigram_features = vectorizer_unigram.get_feature_names_out()

print("--- Unigram Özellikleri ---")
print(f"Özellikler (Kelimeler): {unigram_features}")
print(f"Doküman-Terim Matrisi (Yoğun Format):\n{X_unigram.toarray()}\n")

# --- Bigram Analizi ---
# Metinleri bigram modeline dönüştür ve özellikleri çıkar
X_bigram = vectorizer_bigram.fit_transform(documents)
bigram_features = vectorizer_bigram.get_feature_names_out()

print("--- Bigram Özellikleri ---")
print(f"Özellikler (Kelimeler): {bigram_features}")
print(f"Doküman-Terim Matrisi (Yoğun Format):\n{X_bigram.toarray()}\n")

# --- Trigram Analizi ---
# Metinleri trigram modeline dönüştür ve özellikleri çıkar
X_trigram = vectorizer_trigram.fit_transform(documents)
trigram_features = vectorizer_trigram.get_feature_names_out()

print("--- Trigram Özellikleri ---")
print(f"Özellikler (Kelimeler): {trigram_features}")
print(f"Doküman-Terim Matrisi (Yoğun Format):\n{X_trigram.toarray()}\n")

# Tüm sonuçların özetini yazdırma
print("\n--- Tüm N-gram Sonuçları ---")
print(f"Unigram Özellikleri: {unigram_features}")
print(f"Bigram Özellikleri: {bigram_features}")
print(f"Trigram Özellikleri: {trigram_features}")