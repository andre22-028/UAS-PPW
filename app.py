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
st.markdown("Dashboard ini mengekstraksi kata kunci penting menggunakan perpaduan **Unigram**, **Bigram**, dan algoritma **PageRank**.")

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
    # Simpan versi asli (sedikit dibersihkan dari simbol aneh saja)
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
    
    # Ekstraksi Bigram (Pastikan kedua kata bukan stopword)
    bi_gen = list(ngrams(tokens, 2))
    bigrams = [
        f"{b[0]} {b[1]}" for b in bi_gen 
        if b[0] not in stop_words and b[1] not in stop_words 
        and len(b[0]) > 2 and len(b[1]) > 2
    ]
    
    # Gabungkan unigram menjadi "Kalimat Bersih" untuk demo
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
    st.info("Window size menentukan seberapa jauh jarak antar kata untuk dianggap berhubungan.")

# ===============================
# 5. MAIN LOGIC
# ===============================
if uploaded_file is not None:
    # Ekstraksi Teks dari File
    if uploaded_file.type == "application/pdf":
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            raw_text = "".join([page.get_text() for page in doc])
    else:
        raw_text = uploaded_file.read().decode("utf-8")

    if not raw_text.strip():
        st.error("Dokumen tidak memiliki teks.")
    else:
        # Jalankan Preprocessing
        unigrams, bigrams, cleaned_text, original_text = preprocess_text(raw_text)
        combined_all = unigrams + bigrams
        
        # Inisialisasi Tab
        tab1, tab2, tab3, tab4 = st.tabs([
            "📄 Hasil Preprocessing", 
            "🔢 Matriks Ko-okurensi", 
            "🕸️ Network Graph", 
            "🏆 Skor PageRank (Top 20)"
        ])

        # --- TAB 1: PREPROCESSING & STATISTIK ---
        with tab1:
            st.subheader("Perbandingan Teks")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Teks Asli (Fragmen):**")
                st.text_area("Original", original_text[:1000] + "...", height=200, label_visibility="collapsed")
            with col_b:
                st.markdown("**Teks Hasil Preprocessing (Kalimat Bersih):**")
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
            
            # Buat DataFrame Matriks
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
            st.dataframe(matrix_df, use_container_width=True)

        # --- TAB 3: NETWORK GRAPH ---
        with tab3:
            st.subheader("Representasi Graph Kata")
            G = nx.Graph()
            # Membangun edges
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

        # --- TAB 4: CENTRALITY RESULTS ---
        with tab4:
            st.subheader("Hasil Ekstraksi Kata Kunci Utama")
            # PageRank
            pr_scores = nx.pagerank(G)
            # Degree Centrality (Tambahan minimal 1 sesuai instruksi)
            dc_scores = nx.degree_centrality(G)
            
            final_df = pd.DataFrame({
                "Keyword": list(pr_scores.keys()),
                "PageRank Score": list(pr_scores.values()),
                "Degree Score": [dc_scores[k] for k in pr_scores.keys()]
            }).sort_values(by="PageRank Score", ascending=False).head(20)
            
            final_df.index = range(1, 21)
            st.table(final_df)
            
            # Download Button
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "keyword_results.csv", "text/csv")

else:
    st.info("Silakan unggah dokumen PDF atau TXT untuk memulai ekstraksi.")