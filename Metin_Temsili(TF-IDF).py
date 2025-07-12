import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Veri seti yükleme
df = pd.read_csv("C:/Users/Elif/Desktop/spam.csv", encoding='ISO-8859-1')

# Temizleme fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join([word for word in text.split() if len(word) > 2])
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Temizlenmiş veriyi oluştur
df["cleaned_v2"] = df["v2"].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned_v2"])

# Kelime kümesi ve skorlar
feature_names = vectorizer.get_feature_names_out()
tfidf_score = X.mean(axis=0).A1

# Sonuçların DataFrame'e aktarılması
df_tfidf = pd.DataFrame({"word": feature_names, "tfidf_score": tfidf_score})
df_tfidf_sorted = df_tfidf.sort_values(by="tfidf_score", ascending=False)

# En yüksek skorlu kelimeleri yazdır
print(df_tfidf_sorted.head(10))
