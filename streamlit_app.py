import streamlit as st
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os
import gdown

st.set_page_config(
    page_title="Urdu Poetry Classifier",
    page_icon="📜",
    layout="wide"
)

st.title("📜 Urdu Poetry Genre Classification")
st.markdown("Classify your Urdu poetry into **Ghazal, Nazm, Qita, or Rubai**")

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Download model if not exists
    if not os.path.exists('urdubert_best.pt'):
        st.info("⏳ Downloading model (first time only)...")
        gdown.download('https://drive.google.com/uc?id=1lKeuGxDuEwfArJ9nKbYoutOwBhRtgdb0', 'urdubert_best.pt', quiet=False)
    
    if not os.path.exists('label_encoder.pkl'):
        st.info("⏳ Downloading label encoder...")
        gdown.download('https://drive.google.com/uc?id=1pEtcMgPqj9_hJ3xu7gmHd4LpouZCI7It', 'label_encoder.pkl', quiet=False)
    
    if not os.path.exists('genre_info.pkl'):
        st.info("⏳ Downloading genre info...")
        gdown.download('https://drive.google.com/uc?id=1LZWHOYonAaqbU9HCa7Pt4ojOfBXxwmY4', 'genre_info.pkl', quiet=False)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        'urduhack/roberta-urdu-small',
        num_labels=4,
        ignore_mismatched_sizes=True
    )
    
    model.load_state_dict(torch.load('urdubert_best.pt', map_location=device))
    model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained('urduhack/roberta-urdu-small')
    
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    with open('genre_info.pkl', 'rb') as f:
        genre_info = pickle.load(f)
    
    return model, tokenizer, le, genre_info, device

model, tokenizer, le, GENRE_INFO, device = load_model()

# Sidebar
st.sidebar.title("📚 About Genres")
selected_genre = st.sidebar.radio("Learn about:", le.classes_)
st.sidebar.markdown(f"**{selected_genre}**")
st.sidebar.info(GENRE_INFO[selected_genre])

# Main tabs
tab1, tab2, tab3 = st.tabs(["Single Poem", "Batch Upload", "About"])

with tab1:
    st.subheader("🎯 Classify Your Poem")
    
    poem_text = st.text_area(
        "Enter your Urdu poem:",
        height=250,
        placeholder="یہاں اپنی غزل، نظم یا رباعی لکھیں..."
    )
    
    if st.button("🔍 Classify Poem", type="primary", use_container_width=True):
        if poem_text.strip():
            with st.spinner("Analyzing your poem..."):
                try:
                    enc = tokenizer(
                        poem_text,
                        truncation=True,
                        padding='max_length',
                        max_length=256,
                        return_tensors='pt'
                    )
                    
                    input_ids = enc['input_ids'].to(device)
                    attention_mask = enc['attention_mask'].to(device)
                    
                    with torch.no_grad():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                    
                    predicted_idx = probs.argmax()
                    predicted_genre = le.classes_[predicted_idx]
                    confidence = float(probs[predicted_idx]) * 100
                    
                    lines = [l for l in poem_text.split('\n') if l.strip()]
                    
                    st.success(f"✅ Predicted Genre: **{predicted_genre}**")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    with col2:
                        st.metric("Lines", len(lines))
                    with col3:
                        st.metric("Words", len(poem_text.split()))
                    
                    st.subheader("Genre Scores")
                    scores_df = pd.DataFrame({
                        'Genre': le.classes_,
                        'Confidence': probs * 100
                    })
                    st.bar_chart(scores_df.set_index('Genre'))
                    
                    st.info(f"**{predicted_genre}**: {GENRE_INFO[predicted_genre]}")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a poem!")

with tab2:
    st.subheader("📊 Batch Classification")
    
    uploaded_file = st.file_uploader("Upload CSV (column: 'poem')", type='csv')
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if 'poem' not in df.columns:
            st.error("CSV must have 'poem' column!")
        else:
            if st.button("⚡ Classify All", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(df)} poems..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, poem in enumerate(df['poem']):
                        if not poem.strip():
                            results.append({'Genre': 'Error', 'Confidence': 0})
                            continue
                        
                        enc = tokenizer(
                            poem,
                            truncation=True,
                            padding='max_length',
                            max_length=256,
                            return_tensors='pt'
                        )
                        
                        input_ids = enc['input_ids'].to(device)
                        attention_mask = enc['attention_mask'].to(device)
                        
                        with torch.no_grad():
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                        
                        results.append({
                            'Genre': le.classes_[probs.argmax()],
                            'Confidence': f"{probs.max()*100:.1f}%"
                        })
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    result_df = pd.DataFrame(results)
                    st.dataframe(result_df, use_container_width=True)
                    
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "classified_poems.csv",
                        "text/csv",
                        use_container_width=True
                    )

with tab3:
    st.subheader("ℹ️ About This Project")
    st.markdown('''
    ### Urdu Poetry Genre Classifier
    
    This model classifies Urdu poetry into four main genres:
    
    **📖 Genres:**
    - **Ghazal**: Rhyming couplets with independent meaning
    - **Nazm**: Structured narrative poetry
    - **Qita**: Short poem on single subject
    - **Rubai**: 4-line philosophical verse
    
    **🧠 Model:** UrduBERT (urduhack/roberta-urdu-small)
    **📚 Training Data:** 10,000+ Urdu poems
    **🎯 Accuracy:** ~92%
    **👨‍💻 Built with:** PyTorch, Transformers, Streamlit
    
    ---
    
    **How to use:**
    1. Enter your Urdu poem
    2. Click "Classify Poem"
    3. View predictions
    
    **For bulk classification:**
    1. Upload CSV with 'poem' column
    2. Click "Classify All"
    3. Download results
    ''')

st.markdown("---")
st.caption("🌍 Urdu Poetry Genre Classification | Made with ❤️")