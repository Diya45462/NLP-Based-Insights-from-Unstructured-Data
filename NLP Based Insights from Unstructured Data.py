# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 00:11:59 2026

@author: admin
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

import pandas as pd
df = pd.read_csv('C:/Users/admin/Downloads/AI PROJECT/Reviews.csv.zip')

initial_rows = df.shape[0]

df.drop_duplicates(inplace=True)
df['Text'] = df['Text'].fillna("No content provided")
df.dropna(subset=['Text', 'Score'], inplace=True)

final_rows = df.shape[0]
print(f"Removed {initial_rows - final_rows} rows (duplicates/missing).")
print(f"New dataset size: {final_rows} rows.")

df.reset_index(drop=True, inplace=True)


#text embedding
df['Text'] = df['Text'].fillna("No content")
all_sentences = df['Text'].tolist()
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# 5. Generate embeddings
print(f"Encoding {len(all_sentences)} reviews...")
# show_progress_bar=True helps you see how much time is left
embeddings = model.encode(all_sentences, batch_size=64, show_progress_bar=True)

print(f"Total rows processed: {embeddings.shape[0]}")
print(f"Vector dimensions per row: {embeddings.shape[1]}")

#clustering 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
df = pd.read_csv('C:/Users/admin/Downloads/AI PROJECT/Reviews.csv.zip')

tfidf = TfidfVectorizer(max_features=500, stop_words='english')
vectors = tfidf.fit_transform(df['Text'].fillna(''))
kmeans = KMeans(n_clusters=3, random_state=42)
df['text_cluster'] = kmeans.fit_predict(vectors)

pca = PCA(n_components=2)
coords = pca.fit_transform(vectors.toarray())
import matplotlib.pyplot as plt
plt.scatter(coords[:, 0], coords[:, 1], c=df['text_cluster'], cmap='viridis', alpha=0.5)
plt.title('Clustering Reviews by Text Content')
plt.show()

#semantic search 
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


df = pd.read_csv('C:/Users/admin/Downloads/AI PROJECT/Reviews.csv.zip')

documents = df['Text'].fillna("").head(5000).tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')

doc_embeddings = model.encode(documents, show_progress_bar=True)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings.astype('float32')) # FAISS requires float32


def semantic_search(query, top_k=3):
   
    query_vector = model.encode([query]).astype('float32')
    
   
    distances, indices = index.search(query_vector, top_k)
    
    print(f"\nResults for: '{query}'")
    for i in range(top_k):
        doc_idx = indices[0][i]
        print(f"[{i+1}] Score: {distances[0][i]:.4f}")
        print(f"Text: {documents[doc_idx][:200]}...")
        print("-" * 30)

semantic_search("Are there any reviews mentioning organic ingredients?")
