import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import pymupdf
import nltk
from nltk.corpus import stopwords
from collections import defaultdict, Counter
import numpy as np

# Download resources NLTK
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')

download_nltk_data()

def clean_text(text):
    if not isinstance(text, str): return []
    text = text.replace('◼', '')
    text = re.sub(r'\b(\w+)-\1\b', r'\1', text) 
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = re.sub(r'\b[a-zA-Z]\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    stop_words = set(stopwords.words('indonesian'))
    tokens = text.split()
    cleaned_tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return cleaned_tokens

st.set_page_config(page_title="UAS Keyword Extraction", layout="wide")
st.title("📊 Dashboard Analisis Text Graph")

uploaded_file = st.file_uploader("Upload Paper (PDF atau TXT)", type=['pdf', 'txt'])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
        raw_text = "".join([page.get_text() for page in doc])
    else:
        raw_text = uploaded_file.read().decode("utf-8")

    cleaned_words = clean_text(raw_text)
    
    if cleaned_words:
        # --- TAB PEMISAH ---
        tab1, tab2, tab3, tab4 = st.tabs(["Preprocessing & Frekuensi", "Matriks Ko-okurensi", "Network Graph", "Hasil PageRank"])

        with tab1:
            st.subheader("Hasil Preprocessing")
            st.write(f"Total kata setelah dibersihkan: **{len(cleaned_words)}**")
            st.text_area("Preview Kata:", value=" | ".join(cleaned_words[:100]) + "...", height=150)
            
            st.divider()
            st.subheader("Kata Muncul Terbanyak")
            word_freq = Counter(cleaned_words).most_common(15)
            df_freq = pd.DataFrame(word_freq, columns=['Kata', 'Frekuensi'])
            fig_bar, ax_bar = plt.subplots()
            sns.barplot(x='Frekuensi', y='Kata', data=df_freq, palette='viridis', ax=ax_bar)
            st.pyplot(fig_bar)

        with tab2:
            st.subheader("Matriks Ko-okurensi (Top 15 Kata)")
            # Membangun matriks (mengambil 15 kata terpopuler agar matriks terbaca)
            top_words = [w for w, c in Counter(cleaned_words).most_common(15)]
            matrix = pd.DataFrame(0, index=top_words, columns=top_words)
            
            window_size = 3
            for i, word in enumerate(cleaned_words):
                if word in top_words:
                    start = max(0, i - window_size)
                    end = min(len(cleaned_words), i + window_size + 1)
                    for j in range(start, end):
                        neighbor = cleaned_words[j]
                        if i != j and neighbor in top_words:
                            matrix.loc[word, neighbor] += 1
            st.dataframe(matrix)
            
            # Image tag for context of co-occurrence matrix structure
            # 

        with tab3:
            st.subheader("Co-occurrence Network Graph")
            # Membangun Graph Utama
            G = nx.Graph()
            for i in range(len(cleaned_words)-1):
                w1, w2 = cleaned_words[i], cleaned_words[i+1]
                if G.has_edge(w1, w2):
                    G[w1][w2]['weight'] += 1
                else:
                    G.add_edge(w1, w2, weight=1)

            # Visualisasi Top Nodes
            top_n = st.slider("Jumlah Node yang ditampilkan:", 10, 50, 20)
            top_nodes = [w for w, c in Counter(cleaned_words).most_common(top_n)]
            sub = G.subgraph(top_nodes)
            
            fig_g, ax_g = plt.subplots(figsize=(10, 7))
            pos = nx.kamada_kawai_layout(sub)
            nx.draw(sub, pos, with_labels=True, node_color='orange', edge_color='#D3D3D3', 
                    node_size=800, font_size=9, ax=ax_g)
            st.pyplot(fig_g)
            
            # 

        with tab4:
            st.subheader("Hasil PageRank")
            pagerank_scores = nx.pagerank(G, weight='weight')
            sorted_pr = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
            
            df_pr = pd.DataFrame(sorted_pr, columns=['Kata', 'Skor PageRank'])
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Top 20 Kata dengan Skor Tertinggi:**")
                st.dataframe(df_pr.head(20))
            with col_b:
                st.write("**Statistik Skor:**")
                st.write(df_pr.describe())

    else:
        st.error("File tidak mengandung kata-kata yang valid setelah diproses.")