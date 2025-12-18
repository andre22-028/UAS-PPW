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
    page_title="Keyword Extraction - Dashboard UAS",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Keyword Extraction Berbasis Graph & Centrality")
st.markdown("Dashboard ini mengekstraksi kata kunci penting menggunakan perpaduan **Unigram**, **Bigram**, dan berbagai algoritma **Centrality**.")

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
    # Simpan versi asli
    original_sample = text.replace('◼', '')
    
    # 1. Cleaning & Normalisasi
    clean = original_sample.lower()
    clean = re.sub(r'[^\x00-\x7f]', r'', clean) # Hapus karakter non-ASCII
    clean = clean.translate(str.maketrans('', '', string.punctuation))
    clean = re.sub(r'\d+', '', clean)
    
    # 2. Tokenisasi
    tokens = nltk.word_tokenize(clean)
    
    # 3. Stopword Removal
    stop_words = set(stopwords.words('indonesian'))
    
    # Ekstraksi Unigram
    unigrams = [t for t in tokens if t not in stop_words and len(t) > 2]
    
    # Ekstraksi Bigram
    bi_gen = list(ngrams(tokens, 2))
    bigrams = [
        f"{b[0]} {b[1]}" for b in bi_gen 
        if b[0] not in stop_words and b[1] not in stop_words 
        and len(b[0]) > 2 and len(b[1]) > 2
    ]
    
    cleaned_sentence = " ".join(unigrams)
    return unigrams, bigrams, cleaned_sentence, original_sample

# ===============================
# 4. SIDEBAR & FILE UPLOAD
# ===============================
with st.sidebar:
    st.header("Konfigurasi")
    uploaded_file = st.file_uploader("Upload PDF atau TXT", type=["pdf", "txt"])
    st.divider()
    window_size = st.slider("Graph Window Size", 1, 5, 2)
    st.info("Window size menentukan jarak antar kata untuk dianggap berhubungan dalam graph.")

# ===============================
# 5. MAIN LOGIC
# ===============================
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            raw_text = "".join([page.get_text() for page in doc])
    else:
        raw_text = uploaded_file.read().decode("utf-8")

    if not raw_text.strip():
        st.error("Dokumen tidak memiliki teks.")
    else:
        unigrams, bigrams, cleaned_text, original_text = preprocess_text(raw_text)
        combined_all = unigrams + bigrams
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📄 Hasil Preprocessing", 
            "🔢 Matriks Ko-okurensi", 
            "🕸️ Network Graph", 
            "🏆 Centrality Analysis"
        ])

        # --- TAB 1: PREPROCESSING & STATISTIK ---
        with tab1:
            st.subheader("Perbandingan Teks")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Teks Asli (Fragmen):**")
                st.text_area("Original", original_text[:1000] + "...", height=200, label_visibility="collapsed")
            with col_b:
                st.markdown("**Teks Hasil Preprocessing:**")
                st.text_area("Cleaned", cleaned_text[:1000] + "...", height=200, label_visibility="collapsed")
            
            st.divider()
            st.subheader("Statistik Frekuensi")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Top 15 Unigrams**")
                u_df = pd.DataFrame(Counter(unigrams).most_common(15), columns=["Kata", "Freq"])
                st.bar_chart(u_df.set_index("Kata"))
            with c2:
                st.write("**Top 15 Bigrams**")
                b_df = pd.DataFrame(Counter(bigrams).most_common(15), columns=["Frase", "Freq"])
                st.bar_chart(b_df.set_index("Frase"), color="#ffaa00")

        # --- TAB 2: MATRIKS KO-OKURENSI ---
        with tab2:
            st.subheader("Matriks Hubungan (Top 15 Keywords)")
            top_15 = [item for item, count in Counter(combined_all).most_common(15)]
            matrix_df = pd.DataFrame(0, index=top_15, columns=top_15)
            for i in range(len(combined_all) - 1):
                for j in range(i + 1, min(i + window_size + 1, len(combined_all))):
                    w1, w2 = combined_all[i], combined_all[j]
                    if w1 in top_15 and w2 in top_15:
                        matrix_df.loc[w1, w2] += 1
                        matrix_df.loc[w2, w1] += 1

            fig_m, ax_m = plt.subplots(figsize=(10, 8))
            sns.heatmap(matrix_df, annot=True, cmap="YlGnBu", fmt="d", ax=ax_m)
            st.pyplot(fig_m)

        # --- TAB 3: NETWORK GRAPH ---
        with tab3:
            st.subheader("Representasi Graph Kata")
            G = nx.Graph()
            for i in range(len(combined_all) - 1):
                for j in range(i + 1, min(i + window_size + 1, len(combined_all))):
                    G.add_edge(combined_all[i], combined_all[j])

            num_nodes = st.slider("Jumlah node di graph", 10, 50, 20)
            top_nodes = [n for n, c in Counter(combined_all).most_common(num_nodes)]
            sub = G.subgraph(top_nodes)

            fig_g, ax_g = plt.subplots(figsize=(12, 7))
            pos = nx.kamada_kawai_layout(sub)
            nx.draw(sub, pos, with_labels=True, node_color="#00b4d8", 
                    node_size=2000, font_size=10, edge_color="#dddddd", width=1.5, ax=ax_g)
            st.pyplot(fig_g)

        # --- TAB 4: CENTRALITY RESULTS (DENGAN DEGREE/BETWEENNESS/CLOSENESS) ---
        with tab4:
            st.subheader("Hasil Analisis Centrality")
            
            # Perhitungan Centrality
            pr_scores = nx.pagerank(G)
            dc_scores = nx.degree_centrality(G)
            bc_scores = nx.betweenness_centrality(G)
            cc_scores = nx.closeness_centrality(G)
            
            # Penggabungan ke DataFrame
            centrality_results = []
            for word in pr_scores.keys():
                centrality_results.append({
                    "Keyword": word,
                    "PageRank": pr_scores[word],
                    "Degree": dc_scores[word],
                    "Betweenness": bc_scores[word],
                    "Closeness": cc_scores[word]
                })
            
            full_df = pd.DataFrame(centrality_results)
            
            # Tampilkan Top 20 berdasarkan PageRank
            top_20_df = full_df.sort_values(by="PageRank", ascending=False).head(20)
            top_20_df.index = range(1, 21)
            
            st.markdown("### Top 20 Kata Berdasarkan Berbagai Centrality")
            st.dataframe(top_20_df, use_container_width=True)

            # --- Visualisasi Tambahan ---
            st.divider()
            st.subheader("Visualisasi Metrik")
            metric_choice = st.selectbox("Pilih Metrik untuk Grafik:", ["PageRank", "Degree", "Betweenness", "Closeness"])
            
            fig_v, ax_v = plt.subplots(figsize=(10, 6))
            v_data = full_df.sort_values(by=metric_choice, ascending=False).head(15)
            sns.barplot(x=metric_choice, y="Keyword", data=v_data, palette="magma", ax=ax_v)
            ax_v.set_title(f"Top 15 Keywords - {metric_choice}")
            st.pyplot(fig_v)
            
            # Download
            csv = full_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Full CSV", csv, "centrality_analysis.csv", "text/csv")

else:
    st.info("Silakan unggah dokumen PDF atau TXT untuk memulai ekstraksi.") 