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
Aplikasi ini mengekstraksi kata kunci (Unigram & Bigram) menggunakan algoritma **PageRank** dan **Degree Centrality** berdasarkan struktur graf kata.
""")

# ===============================
# 2. DOWNLOAD RESOURCE NLTK
# ===============================
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

download_nltk_data()

# ===============================
# 3. FUNGSI PREPROCESSING TEKS
# ===============================
def preprocess_text(text):
    # Bersihkan simbol aneh dan karakter non-ascii
    text = text.replace('◼', '')
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    
    # Normalisasi: Case folding & hapus angka/tanda baca
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    
    # Tokenisasi
    tokens = nltk.word_tokenize(text)
    
    # Filter Stopwords (Bahasa Indonesia)
    stop_words = set(stopwords.words('indonesian'))
    
    # List Unigram (Hanya kata yang bukan stopword dan panjang > 2)
    unigrams = [t for t in tokens if t not in stop_words and len(t) > 2]
    
    # List Bigram (Dua kata berurutan yang keduanya bukan stopword)
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
    st.info("Aplikasi ini akan memproses Unigram dan Bigram secara bersamaan dalam Graph.")

# ===============================
# 5. MAIN LOGIC
# ===============================
if uploaded_file is not None:
    # Ekstraksi Teks
    if uploaded_file.type == "application/pdf":
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            raw_text = ""
            for page in doc:
                raw_text += page.get_text()
    else:
        raw_text = uploaded_file.read().decode("utf-8")

    if raw_text.strip() == "":
        st.error("Dokumen kosong atau tidak terbaca.")
    else:
        # Preprocessing
        unigrams, bigrams = preprocess_text(raw_text)
        # Gabungkan untuk Graph (Sesuai instruksi soal: Unigram + Bigram)
        combined_keywords = unigrams + bigrams
        
        # --- TAB MENU ---
        tab_stats, tab_graph, tab_result = st.tabs([
            "📊 Preprocessing & Statistik", 
            "🕸️ Visualisasi Graph", 
            "🏆 Centrality Score (Top 20)"
        ])

        # TAB 1: STATISTIK TERPISAH
        with tab_stats:
            st.subheader("Hasil Preprocessing (Pemisahan)")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top 15 Unigrams**")
                df_u = pd.DataFrame(Counter(unigrams).most_common(15), columns=["Kata", "Freq"])
                st.bar_chart(data=df_u, x="Kata", y="Freq")
                st.dataframe(df_u, use_container_width=True)

            with col2:
                st.write("**Top 15 Bigrams**")
                df_b = pd.DataFrame(Counter(bigrams).most_common(15), columns=["Frase", "Freq"])
                st.bar_chart(data=df_b, x="Frase", y="Freq", color="#ffaa00")
                st.dataframe(df_b, use_container_width=True)

        # TAB 2: PEMBANGUNAN GRAPH
        with tab_graph:
            st.subheader("Network Graph (Co-occurrence)")
            
            # Membangun Graph menggunakan NetworkX
            G = nx.Graph()
            # Windowing: Hubungkan kata yang muncul berdekatan
            window_size = 2
            for i in range(len(combined_keywords) - 1):
                for j in range(i + 1, min(i + window_size + 1, len(combined_keywords))):
                    w1, w2 = combined_keywords[i], combined_keywords[j]
                    if G.has_edge(w1, w2):
                        G[w1][w2]['weight'] += 1
                    else:
                        G.add_edge(w1, w2, weight=1)

            num_nodes = st.slider("Jumlah node untuk ditampilkan di graph", 10, 50, 25)
            top_nodes = [n for n, c in Counter(combined_keywords).most_common(num_nodes)]
            sub = G.subgraph(top_nodes)

            # Gambar Graph
            fig, ax = plt.subplots(figsize=(12, 8))
            pos = nx.kamada_kawai_layout(sub)
            
            # Node size berdasarkan degree
            d = dict(sub.degree)
            nx.draw(sub, pos, 
                    with_labels=True, 
                    node_color="#00b4d8", 
                    node_size=[v * 500 for v in d.values()],
                    font_size=9, 
                    edge_color="#cccccc",
                    width=1.5,
                    alpha=0.8,
                    ax=ax)
            st.pyplot(fig)

        # TAB 3: CENTRALITY (PAGERANK & DEGREE)
        with tab_result:
            st.subheader("Top 20 Keywords Berdasarkan Centrality")
            
            # Perhitungan Centrality
            # 1. PageRank (Wajib)
            pagerank_scores = nx.pagerank(G, weight='weight')
            # 2. Degree Centrality (Tambahan minimal 1)
            degree_scores = nx.degree_centrality(G)
            
            # Buat DataFrame hasil
            df_final = pd.DataFrame({
                "Keyword": list(pagerank_scores.keys()),
                "PageRank Score": list(pagerank_scores.values()),
                "Degree Score": [degree_scores[k] for k in pagerank_scores.keys()]
            })
            
            # Ambil Top 20 berdasarkan PageRank
            df_top_20 = df_final.sort_values(by="PageRank Score", ascending=False).head(20)
            df_top_20.index = range(1, 21) # Reset index jadi 1-20

            st.table(df_top_20)
            
            # Download CSV
            csv = df_top_20.to_csv(index=False).encode('utf-8')
            st.download_button("Download Tabel (CSV)", csv, "top_20_keywords.csv", "text/csv")

else:
    st.warning("Silakan upload file PDF atau TXT melalui sidebar untuk memulai.")

# ===============================
# FOOTER
# ===============================
st.divider()
st.caption("UAS Keyword Extraction - Powered by Streamlit & NetworkX")