import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from collections import Counter
import numpy as np

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="UAS Keyword Extraction",
    layout="wide"
)

st.title("📊 Dashboard Analisis Text Graph")

# ===============================
# DOWNLOAD RESOURCE NLTK
# ===============================
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')

download_nltk_data()

# ===============================
# FUNGSI PREPROCESSING TEKS
# ===============================
def clean_text(text):
    if not isinstance(text, str):
        return []

    # Normalisasi teks
    text = text.replace('◼', '')
    text = re.sub(r'\b(\w+)-\1\b', r'\1', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = re.sub(r'\b[a-zA-Z]\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    stop_words = set(stopwords.words('indonesian'))
    tokens = text.split()

    cleaned_tokens = [
        token for token in tokens
        if token not in stop_words and len(token) > 1
    ]

    return cleaned_tokens

# ===============================
# UPLOAD FILE
# ===============================
uploaded_file = st.file_uploader(
    "Upload Dokumen (PDF atau TXT)",
    type=["pdf", "txt"]
)

# ===============================
# PROSES FILE
# ===============================
if uploaded_file is not None:

    # ----- PDF -----
    if uploaded_file.type == "application/pdf":
        doc = fitz.open(
            stream=uploaded_file.read(),
            filetype="pdf"
        )
        raw_text = ""
        for page in doc:
            raw_text += page.get_text()

    # ----- TXT -----
    else:
        raw_text = uploaded_file.read().decode("utf-8")

    cleaned_words = clean_text(raw_text)

    if len(cleaned_words) > 0:

        tab1, tab2, tab3, tab4 = st.tabs([
            "Preprocessing & Frekuensi",
            "Matriks Ko-okurensi",
            "Network Graph",
            "PageRank"
        ])

        # ===============================
        # TAB 1: PREPROCESSING
        # ===============================
        with tab1:
            st.subheader("Hasil Preprocessing")
            st.write(f"Total kata setelah dibersihkan: **{len(cleaned_words)}**")

            st.text_area(
                "Preview Kata",
                value=" | ".join(cleaned_words[:100]) + "...",
                height=150
            )

            st.divider()

            st.subheader("15 Kata Paling Sering Muncul")
            word_freq = Counter(cleaned_words).most_common(15)
            df_freq = pd.DataFrame(word_freq, columns=["Kata", "Frekuensi"])

            fig_bar, ax_bar = plt.subplots()
            sns.barplot(
                data=df_freq,
                x="Frekuensi",
                y="Kata",
                ax=ax_bar
            )
            st.pyplot(fig_bar)

        # ===============================
        # TAB 2: KO-OKURENSI
        # ===============================
        with tab2:
            st.subheader("Matriks Ko-okurensi (Top 15 Kata)")

            top_words = [
                w for w, _ in Counter(cleaned_words).most_common(15)
            ]

            matrix = pd.DataFrame(
                0,
                index=top_words,
                columns=top_words
            )

            window_size = 3

            for i, word in enumerate(cleaned_words):
                if word in top_words:
                    start = max(0, i - window_size)
                    end = min(len(cleaned_words), i + window_size + 1)

                    for j in range(start, end):
                        if i != j:
                            neighbor = cleaned_words[j]
                            if neighbor in top_words:
                                matrix.loc[word, neighbor] += 1

            st.dataframe(matrix)

        # ===============================
        # TAB 3: NETWORK GRAPH
        # ===============================
        with tab3:
            st.subheader("Co-occurrence Network Graph")

            G = nx.Graph()

            for i in range(len(cleaned_words) - 1):
                w1 = cleaned_words[i]
                w2 = cleaned_words[i + 1]

                if G.has_edge(w1, w2):
                    G[w1][w2]["weight"] += 1
                else:
                    G.add_edge(w1, w2, weight=1)

            top_n = st.slider(
                "Jumlah node ditampilkan",
                min_value=10,
                max_value=50,
                value=20
            )

            top_nodes = [
                w for w, _ in Counter(cleaned_words).most_common(top_n)
            ]

            subgraph = G.subgraph(top_nodes)

            fig_net, ax_net = plt.subplots(figsize=(10, 7))
            pos = nx.kamada_kawai_layout(subgraph)

            nx.draw(
                subgraph,
                pos,
                with_labels=True,
                node_size=800,
                font_size=9,
                node_color="orange",
                edge_color="#D3D3D3",
                ax=ax_net
            )

            st.pyplot(fig_net)

        # ===============================
        # TAB 4: PAGERANK
        # ===============================
        with tab4:
            st.subheader("Hasil PageRank")

            pagerank_scores = nx.pagerank(G, weight="weight")
            sorted_pr = sorted(
                pagerank_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            df_pr = pd.DataFrame(
                sorted_pr,
                columns=["Kata", "Skor PageRank"]
            )

            col1, col2 = st.columns(2)

            with col1:
                st.write("Top 20 Kata dengan Skor Tertinggi")
                st.dataframe(df_pr.head(20))

            with col2:
                st.write("Statistik Skor PageRank")
                st.dataframe(df_pr.describe())

    else:
        st.error("Dokumen tidak mengandung kata valid setelah preprocessing.")
