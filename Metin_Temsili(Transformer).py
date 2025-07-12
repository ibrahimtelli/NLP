import pandas as pd
import re
import torch
from transformers import BertTokenizer, BertModel
import nltk
from nltk.corpus import stopwords

# nltk yüklemeleri
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Veri setini yükle
df = pd.read_csv("C:/Users/Elif/Desktop/IMDB Dataset.csv")

# Temizleme fonksiyonu
def clean_text(text): 
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join([word for word in text.split() if len(word) > 2])
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Temizlenmiş metinleri oluştur
df["cleaned_review"] = df["review"].apply(clean_text)

# İlk 20 metni al
texts = df["cleaned_review"][:20].tolist()

# BERT tokenizer ve model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Metinleri tokenize et
encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)

# Modelden [CLS] token embedding'lerini al
with torch.no_grad():
    model_output = model(**encoded_input)

embeddings = model_output.last_hidden_state[:, 0, :].numpy()

# Embeddingleri DataFrame olarak tut
df_bert = pd.DataFrame(embeddings)
