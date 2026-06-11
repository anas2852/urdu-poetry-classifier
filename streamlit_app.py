import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import gdown
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="Urdu Poetry Classifier",
    page_icon="📜",
    layout="wide"
)

st.title("📜 Urdu Poetry Genre Classification")
st.markdown("Classify your Urdu poetry into **Ghazal, Nazm, Qita, or Rubai** using individual deep-learning components or a hybrid ensemble matrix.")

# =====================================================================
# 1. REQUIRED GRU CLASSES & FEATURE EXTRACTORS
# =====================================================================

class UrduWordTokenizer:
    def __init__(self, max_vocab=10000, max_len=150):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}

    def encode(self, text):
        ids = [self.word2idx.get(w, 1) for w in text.split()[:self.max_len]]
        ids += [0] * (self.max_len - len(ids))
        return ids

class GRUAttentionStructural(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256, num_classes=4, struct_dim=7, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=dropout)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.struct_fc = nn.Sequential(nn.Linear(struct_dim, 32), nn.ReLU(), nn.Linear(32, 32))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 2 + 32, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x, struct):
        x = self.dropout(self.embedding(x))
        out, _ = self.gru(x)
        weights = torch.softmax(self.attention(out), dim=1)
        context = (weights * out).sum(dim=1)
        struct_out = self.struct_fc(struct)
        combined = torch.cat([context, struct_out], dim=1)
        return self.classifier(combined)

def extract_structural_features(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines: return [0] * 7
    line_count = len(lines)
    word_counts = [len(l.split()) for l in lines]
    avg_words_per_line = np.mean(word_counts) if lines else 0
    std_words_per_line = np.std(word_counts) if lines else 0
    last_words = [l.split()[-1] if l.split() else '' for l in lines]
    last_3chars = [w[-3:] if len(w) >= 3 else w for w in last_words]
    
    even_endings = last_3chars[1::2]
    even_rhyme_ratio = Counter(even_endings).most_common(1)[0][1] / len(even_endings) if even_endings else 0.0
    all_most_common = Counter(last_3chars).most_common(1)[0][1] if last_3chars else 0
    all_rhyme_ratio = all_most_common / len(last_3chars) if last_3chars else 0.0
    even_last_words = last_words[1::2]
    radif_ratio = Counter(even_last_words).most_common(1)[0][1] / len(even_last_words) if even_last_words else 0.0

    return [
        line_count / 40.0, avg_words_per_line / 15.0, std_words_per_line / 5.0,
        even_rhyme_ratio, all_rhyme_ratio, radif_ratio, 1.0 if line_count == 4 else 0.0
    ]

# =====================================================================
# 2. CACHED INFRASTRUCTURE RESOURCING (Downloads & Initializations)
# =====================================================================

@st.cache_resource
def load_all_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists('urdubert_best.pt'):
        st.info("⏳ Downloading UrduBERT Model weights...")
        gdown.download('https://drive.google.com/uc?id=1NbAVL82t_1Vw5rwTsJlbUeV_JfgKZDdZ', 'urdubert_best.pt', quiet=False)
        
    if not os.path.exists('gru_structural_best.pt'):
        st.info("⏳ Downloading BiGRU Structural Model weights...")
        gdown.download('https://drive.google.com/uc?id=1C4rs4OlxOUYpobPjKg3Nj1EcnJ5a_jkm', 'gru_structural_best.pt', quiet=False)
        
    if not os.path.exists('gru_tokenizer.pkl'):
        st.info("⏳ Downloading GRU custom tokenizer...")
        gdown.download('https://drive.google.com/uc?id=1AWfKNG2CdSAZFd4THQcKOu_m432K1klM', 'gru_tokenizer.pkl', quiet=False)
        
    if not os.path.exists('label_encoder.pkl'):
        st.info("⏳ Downloading metadata label map...")
        gdown.download('https://drive.google.com/uc?id=1wT08vU7QU5A9luxGSwuzVMISvnKsAaSq', 'label_encoder.pkl', quiet=False)
        
    if not os.path.exists('genre_info.pkl'):
        st.info("⏳ Downloading metadata dictionary info...")
        gdown.download('https://drive.google.com/uc?id=1yWzZstnuEqKodLR6Ta3utaeLdIWe9kgI', 'genre_info.pkl', quiet=False)

    # ── Load Core Transformers Component ──
    bert_tokenizer = AutoTokenizer.from_pretrained('urduhack/roberta-urdu-small')
    bert_model = AutoModelForSequenceClassification.from_pretrained('urduhack/roberta-urdu-small', num_labels=4, ignore_mismatched_sizes=True)
    bert_model.load_state_dict(torch.load('urdubert_best.pt', map_location=device))
    bert_model.to(device).eval()
    
    # ── Load Custom BiGRU Component ──
    with open('gru_tokenizer.pkl', 'rb') as f:
        gru_tokenizer = pickle.load(f)
        
    gru_weights = torch.load('gru_structural_best.pt', map_location=device, weights_only=False)

    if not isinstance(gru_weights, dict):
        gru_model = gru_weights
    else:
        try:
            emb_key = next(k for k in gru_weights.keys() if 'embedding.weight' in k)
            gru_key = next(k for k in gru_weights.keys() if 'gru.weight_ih_l0' in k)
            
            extracted_vocab_size = gru_weights[emb_key].shape[0]
            extracted_embed_dim = gru_weights[emb_key].shape[1]
            extracted_hidden_dim = gru_weights[gru_key].shape[0] // 3
        except StopIteration:
            extracted_vocab_size = 10000
            extracted_embed_dim = 128
            extracted_hidden_dim = 256
            
        gru_model = GRUAttentionStructural(
            vocab_size=extracted_vocab_size,
            embed_dim=extracted_embed_dim,
            hidden_dim=extracted_hidden_dim,
            num_classes=4
        )
        
        clean_weights = {}
        for k, v in gru_weights.items():
            new_key = k.split('embedding.')[-1] if 'embedding.' in k else k
            if k.startswith('module.'): new_key = k[7:]
            elif k.startswith('_orig_mod.'): new_key = k[10:]
            clean_weights[new_key] = v
            
        gru_model.load_state_dict(clean_weights, strict=False)

    gru_model.to(device).eval()
    
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('genre_info.pkl', 'rb') as f:
        genre_info = pickle.load(f)
        
    return device, le, genre_info, bert_model, bert_tokenizer, gru_model, gru_tokenizer

device, le, GENRE_INFO, bert_model, bert_tokenizer, gru_model, gru_tokenizer = load_all_models()

# =====================================================================
# 3. INTERFACE MANIPULATION LOGIC
# =====================================================================

st.sidebar.title("🤖 Engine Configurations")
model_choice = st.sidebar.selectbox(
    "Select Target Model Architecture:",
    ["UrduBERT (Transformer)", "BiGRU + Structural Features", "Ensemble Combo (Hybrid Strategy)"]
)

st.sidebar.markdown("---")
st.sidebar.title("📚 About Genres")
selected_genre = st.sidebar.radio("Learn about:", le.classes_)
st.sidebar.markdown(f"**{selected_genre}**")
st.sidebar.info(GENRE_INFO[selected_genre])

def execute_prediction(poem_raw_text):
    struct_feats = extract_structural_features(poem_raw_text)
    gru_probs, bert_probs = None, None
    
    with torch.no_grad():
        if model_choice in ["BiGRU + Structural Features", "Ensemble Combo (Hybrid Strategy)"]:
            encoded_gru = gru_tokenizer.encode(poem_raw_text)
            
            vocab_limit = gru_model.embedding.num_embeddings
            encoded_gru = [idx if idx < vocab_limit else 1 for idx in encoded_gru]
            
            t_x = torch.tensor([encoded_gru], dtype=torch.long).to(device)
            t_sf = torch.tensor([struct_feats], dtype=torch.float).to(device)
            gru_logits = gru_model(t_x, t_sf)
            gru_probs = torch.softmax(gru_logits, dim=1).cpu().numpy()[0]
            
        if model_choice in ["UrduBERT (Transformer)", "Ensemble Combo (Hybrid Strategy)"]:
            encoded_bert = bert_tokenizer(poem_raw_text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
            b_ids = encoded_bert['input_ids'].to(device)
            b_mask = encoded_bert['attention_mask'].to(device)
            bert_logits = bert_model(input_ids=b_ids, attention_mask=b_mask).logits
            bert_probs = torch.softmax(bert_logits, dim=1).cpu().numpy()[0]
            
    if model_choice == "UrduBERT (Transformer)":
        return bert_probs
    elif model_choice == "BiGRU + Structural Features":
        return gru_probs
    else:
        # ── 🎛️ DYNAMIC GATING MECHANISM INSIDE ENSEMBLE ENGINE ──
        raw_lines = [line.strip() for line in poem_raw_text.split('\n') if line.strip()]
        
        if len(raw_lines) == 4:
            # 4-Line Text structural routing matrix (Favors BiGRU's structural boundary checks)
            return (0.20 * bert_probs) + (0.80 * gru_probs)
        else:
            # Traditional form routing matrix (Favors UrduBERT's high-efficiency semantics context)
            return (0.90 * bert_probs) + (0.10 * gru_probs)
    
# Tabs Layout Matrix
tab1, tab2, tab3 = st.tabs(["Single Poem", "Batch Upload", "About"])

with tab1:
    st.subheader("🎯 Classify Your Poem")
    poem_text = st.text_area("Enter your Urdu poem:", height=250, placeholder="یہاں اپنی غزل، نظم یا رباعی لکھیں...")
    
    if st.button("🔍 Classify Poem", type="primary", use_container_width=True):
        if poem_text.strip():
            with st.spinner("Analyzing structural forms and semantics..."):
                try:
                    probs = execute_prediction(poem_text)
                    predicted_idx = probs.argmax()
                    predicted_genre = le.classes_[predicted_idx]
                    confidence = float(probs[predicted_idx]) * 100
                    lines = [l for l in poem_text.split('\n') if l.strip()]
                    
                    st.success(f"✅ Predicted Genre: **{predicted_genre}** _({model_choice} Engine)_")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    with col2:
                        st.metric("Lines", len(lines))
                    with col3:
                        st.metric("Words", len(poem_text.split()))
                    
                    st.subheader("Genre Scores Matrix")
                    scores_df = pd.DataFrame({'Genre': le.classes_, 'Confidence': probs * 100})
                    st.bar_chart(scores_df.set_index('Genre'))
                    
                except Exception as e:
                    st.error(f"Execution Error Encountered: {e}")
        else:
            st.warning("Please enter a valid poem sequence first.")

with tab2:
    st.subheader("📊 Batch Classification")
    uploaded_file = st.file_uploader("Upload CSV (column: 'poem')", type='csv')
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'poem' not in df.columns:
            st.error("CSV must contain a column named exactly 'poem'!")
        else:
            if st.button("⚡ Classify All Rows", type="primary", use_container_width=True):
                with st.spinner(f"Batch processing {len(df)} records..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, poem in enumerate(df['poem']):
                        if not str(poem).strip() or pd.isna(poem):
                            results.append({'Genre': 'Data Missing/Error', 'Confidence': "0.0%"})
                            continue
                        
                        probs = execute_prediction(str(poem))
                        results.append({
                            'Genre': le.classes_[probs.argmax()],
                            'Confidence': f"{probs.max()*100:.1f}%"
                        })
                        progress_bar.progress((idx + 1) / len(df))
                    
                    result_df = pd.concat([df, pd.DataFrame(results)], axis=1)
                    st.dataframe(result_df, use_container_width=True)
                    
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Results", csv, "classified_poems.csv", "text/csv", use_container_width=True)

with tab3:
    st.subheader("ℹ️ About This Project")
    st.markdown(f'''
    ### Multi-Engine Urdu Poetry Genre Classification Pipeline
    
    This app leverages text embeddings along with manual stylistic rule injection to evaluate patterns.
    
    **🧠 Active Engine:** `{model_choice}`
    - **UrduBERT**: Optimized for lexical intent and emotional semantic profiles.
    - **BiGRU + Structural Features**: Tracks verse-level boundaries, end-word rhyming weights (*Qafia*), and structural repeating strings (*Radif*).
    - **Ensemble Combo**: Calculates dynamic probability soft-voting combinations between both neural graphs for stable boundary performance.
    ''')

st.markdown("---")
st.caption("🌍 Urdu Poetry Genre Classification Dashboard | Platform Complete")