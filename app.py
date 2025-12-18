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
from nltk.util import ngrams
from collections import Counter

# ===============================
# 1. KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Keyword Extraction - Graph Based",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Keyword Extraction Berbasis Graph & Centrality")
st.markdown("""
Aplikasi ini mengekstraksi kata kunci (**Unigram & Bigram**) menggunakan algoritma **PageRank** dan **Degree Centrality** berdasarkan struktur graf kata (Co-occurrence).
""")

# ===============================
# 2. DOWNLOAD RESOURCE NLTK
# ===============================
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')
    except:
        pass

download_nltk_data()

# ===============================
# 3. FUNGSI PREPROCESSING TEKS
# ===============================
def preprocess_text(text):
    # Bersihkan simbol khusus
    text = text.replace('◼', '')
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    
    # Normalisasi
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('indonesian'))
    
    # 1. Ekstraksi Unigram
    unigrams = [t for t in tokens if t not in stop_words and len(t) > 2]
    
    # 2. Ekstraksi Bigram
    bi_grams_gen = list(ngrams(tokens, 2))
    bigrams = [
        f"{b[0]} {b[1]}" for b in bi_grams_gen 
        if b[0] not in stop_words and b[1] not in stop_words 
        and len(b[0]) > 2 and len(b[1]) > 2
    ]
    
    return unigrams, bigrams

# ===============================
# 4. SIDEBAR - INPUT FILE
# ===============================
with st.sidebar:
    st.header("Upload Dokumen")
    uploaded_file = st.file_uploader("Pilih PDF atau TXT", type=["pdf", "txt"])
    st.divider()
    st.info("Aplikasi akan membangun graph berdasarkan hubungan antar kata (Unigram + Bigram).")

# ===============================
# 5. MAIN LOGIC
# ===============================
if uploaded_file is not None:
    # Proses Ekstraksi Teks
    if uploaded_file.type == "application/pdf":
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            raw_text = "".join([page.get_text() for page in doc])
    else:
        raw_text = uploaded_file.read().decode("utf-8")

    if raw_text.strip() == "":
        st.error("Dokumen kosong.")
    else:
        # Preprocessing
        unigrams, bigrams = preprocess_text(raw_text)
        combined_keywords = unigrams + bigrams # Digabung untuk Graph
        
        tab_stats, tab_matrix, tab_graph, tab_result = st.tabs([
            "📊 Statistik", 
            "🔢 Matriks Co-occurrence",
            "🕸️ Visualisasi Graph", 
            "🏆 Top 20 Keywords"
        ])

        # --- TAB 1: STATISTIK ---
        with tab_stats:
            st.subheader("Frekuensi Kata & Frase")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Top 15 Unigrams**")
                df_u = pd.DataFrame(Counter(unigrams).most_common(15), columns=["Kata", "Freq"])
                st.bar_chart(data=df_u, x="Kata", y="Freq")
            with c2:
                st.write("**Top 15 Bigrams**")
                df_b = pd.DataFrame(Counter(bigrams).most_common(15), columns=["Frase", "Freq"])
                st.bar_chart(data=df_b, x="Frase", y="Freq", color="#ffaa00")

        # --- TAB 2: MATRIKS CO-OCCURRENCE ---
        with tab_matrix:
            st.subheader("Matriks Ko-okurensi (Top 15 Keywords)")
            top_15_all = [item for item, count in Counter(combined_keywords).most_common(15)]
            
            # Membangun Matriks
            matrix = pd.DataFrame(0, index=top_15_all, columns=top_15_all)
            window = 2
            for i in range(len(combined_keywords) - 1):
                for j in range(i + 1, min(i + window + 1, len(combined_keywords))):
                    w1, w2 = combined_keywords[i], combined_keywords[j]
                    if w1 in top_15_all and w2 in top_15_all:
                        matrix.loc[w1, w2] += 1
                        matrix.loc[w2, w1] += 1

            fig_m, ax_m = plt.subplots(figsize=(10, 8))
            sns.heatmap(matrix, annot=True, cmap="YlGnBu", ax=ax_m)
            st.pyplot(fig_m)
            st.dataframe(matrix)

        # --- TAB 3: GRAPH ---
        with tab_graph:
            st.subheader("Network Graph")
            G = nx.Graph()
            for i in range(len(combined_keywords) - 1):
                w1, w2 = combined_keywords[i], combined_keywords[i+1]
                if G.has_edge(w1, w2):
                    G[w1][w2]['weight'] += 1
                else:
                    G.add_edge(w1, w2, weight=1)

            num_nodes = st.slider("Tampilkan jumlah node", 10, 50, 20)
            top_nodes = [n for n, c in Counter(combined_keywords).most_common(num_nodes)]
            sub = G.subgraph(top_nodes)
            
            fig_g, ax_g = plt.subplots(figsize=(12, 8))
            pos = nx.spring_layout(sub, k=1)
            nx.draw(sub, pos, with_labels=True, node_color="#00b4d8", 
                    node_size=1500, font_size=9, edge_color="#cccccc", ax=ax_g)
            st.pyplot(fig_g)

        # --- TAB 4: HASIL AKHIR ---
        with tab_result:
            st.subheader("Top 20 Keywords (PageRank & Degree)")
            pr = nx.pagerank(G, weight='weight')
            dc = nx.degree_centrality(G)
            
            df_res = pd.DataFrame({
                "Keyword": list(pr.keys()),
                "PageRank": list(pr.values()),
                "Degree Centrality": [dc[k] for k in pr.keys()]
            }).sort_values(by="PageRank", ascending=False).head(20)
            
            df_res.index = range(1, 21)
            st.table(df_res)
            
            csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "keywords.csv", "text/csv")

else:
    st.info("Silakan unggah file dokumen untuk memulai analisis.")