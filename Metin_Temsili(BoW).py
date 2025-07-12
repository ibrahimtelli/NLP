import pandas as pd 
import numpy as np
import re #metin temizleme
from sklearn.feature_extraction.text import CountVectorizer #BoW kütüphanesi
from collections import Counter #frekans hesaplama
from nltk.corpus import stopwords

df = pd.read_csv("C:/Users/Elif/Desktop/IMDB Dataset.csv")
stop_words = set(stopwords.words('english'))

#metin verileri

documents = df["review"]
labels = df["sentiment"]

#veri temizleme 

def clean_text(text): 
    
    #büyük-küçük harf
    text = text.lower()
    
    #rakamları temizleme
    text = re.sub(r"\d+", "", text)
    
    #ozel karakterlerin kaldırılması
    text = re.sub(r"[^\w\s]", "", text)
    
    #kısa kelimelerin temizleme
    text = " ".join([word for word in text.split() if len(word) > 2])
    
    #stop words cıkarma
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text #temizlenmiş texti return etme

cleaned_doc = [clean_text(row) for row in documents]

#BoW Yöntemi

#Vectorizer Tanımlama
vectorizer = CountVectorizer()

#Metni sayısal hale getir
X = vectorizer.fit_transform(cleaned_doc[:75])

#Kelime Kümesi
feature_names = vectorizer.get_feature_names_out()

#Vektör Temsili
vektor_temsili2 = X.toarray()
df_bow = pd.DataFrame(vektor_temsili2, columns = feature_names)

#Kelime Frekanlarını Görme
word_counts = X.sum(axis=0).A1
word_freq = dict(zip(feature_names, word_counts))

#İlk 10 kelime
most_common_10_words = Counter(word_freq).most_common(5)
print(f"most_common_10_words: {most_common_10_words}")






