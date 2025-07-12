#Kütüphaneler
import pandas as pd 
import re 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

#Veri seti yükleme
df = pd.read_csv("C:/Users/Elif/Desktop/IMDB Dataset.csv")
documents = df["review"]
stop_words = set(stopwords.words('english'))

#Metin temizleme
def clean_text(text):
    text = text.lower()     # Metindeki tüm harfleri küçük harfe dönüştürür.
    text = re.sub(r'\d+', "", text)     # Metindeki tüm sayıları (0-9) boşluk karakteriyle değiştirir. (Örn: "123" -> " ") 
    text = re.sub(r'[^a-zA-Z0-9\s]',"", text)     # Metindeki alfabe, sayı ve alt çizgi dışındaki tüm özel karakterleri boşluk karakteriyle değiştirir.
    text = " ".join([word for word in text.split() if len(word) > 2]) # Metni kelimelere ayırır, ardından uzunluğu 2 karakterden büyük olan kelimeleri seçer ve bu kelimeleri tekrar tek bir dize olarak birleştirir.
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text # Temizlenmiş metni döndürür

cleaned_documents = [clean_text(doc) for doc in documents] # temizlenmiş belgeleri 'cleaned_documents' listesine kaydeder.

#Metin Tokenizasyonu
tokenized_documents = [simple_preprocess(doc) for doc in cleaned_documents]

# word2vec modeli tanimla
model = Word2Vec(sentences=tokenized_documents, vector_size=50, window=5, min_count=1, sg=0)
word_vectors = model.wv

words = list(word_vectors.index_to_key)[:500]
vectors = [word_vectors[word] for word in words]

# clustering KMeans K=2
kmeans = KMeans(n_clusters=2)
kmeans.fit(vectors)
clusters = kmeans.labels_ # 0, 1

# PCA 50 -> 2
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

# 2 boyutlu bir görselleştirme
plt.figure()
plt.scatter(reduced_vectors[:,0], reduced_vectors[:,1], c = clusters, cmap = "viridis")

centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0], centers[:,1], c="red", marker = "X", s = 150, label = "Center")
plt.legend()

# figure üzerine kelimelerin eklenmesi
for i, word in enumerate(words):
    plt.text(reduced_vectors[i,0], reduced_vectors[i,1], word, fontsize = 7)

plt.title("Word2Vec")