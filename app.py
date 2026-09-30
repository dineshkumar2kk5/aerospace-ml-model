import streamlit as st
import json
import time
from semantic_explainer import SemanticExplainer

st.set_page_config(
    page_title="AeroBeacon - Semantic Meteorology AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Luxury / Aviation Aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #00E5FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #90CAF9;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .concept-card {
        background: #0E1626;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .chapter-badge {
        background: rgba(30, 136, 229, 0.2);
        color: #64B5F6;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .answer-box {
        background: rgba(46, 125, 50, 0.15);
        border-left: 4px solid #4CAF50;
        padding: 10px 14px;
        border-radius: 4px;
        color: #A5D6A7;
        font-weight: 600;
        margin: 10px 0;
    }
    .explanation-box {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #00B0FF;
        padding: 10px 14px;
        border-radius: 4px;
        color: #E2E8F0;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_semantic_engine():
    return SemanticExplainer()

with st.spinner("✈️ Loading Dense Semantic ML Model & Cross-Encoder..."):
    engine = get_semantic_engine()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/airplane-take-off.png", width=70)
    st.title("AeroBeacon AI")
    st.caption("DGCA / CPL / ATPL Aviation Meteorology")
    
    st.markdown("---")
    st.subheader("Model Specifications")
    st.markdown("""
    - **Bi-Encoder**: `all-MiniLM-L6-v2` (384-D)
    - **Cross-Encoder**: `ms-marco-MiniLM-L-6-v2`
    - **Knowledge Base**: IC Joshi 7th Ed. (659 Q&As)
    - **Engine Type**: True Latent ML (Zero Regex)
    """)
    
    st.markdown("---")
    st.subheader("Sample Queries")
    sample_queries = [
        "Why is the tropopause thicker at the equator?",
        "When flying from high to low pressure, what happens to altimeter?",
        "Which ice accretion is glassy and hard to break off?",
        "What is a microburst and how large is its diameter?",
        "What causes clear air turbulence near jet streams?",
        "Where on Earth is Coriolis force maximum?"
    ]
    
    selected_sample = st.selectbox("Choose a sample query:", ["Select an example..."] + sample_queries)

# Main UI
st.markdown('<div class="main-title">AeroBeacon - Semantic Meteorology ML</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Neural Vector Retrieval + Cross-Attention Reranking for Aviation Questions</div>', unsafe_allow_html=True)

# Search Input
default_text = selected_sample if selected_sample != "Select an example..." else ""
query_input = st.text_input("Ask any aviation meteorology question or concept:", value=default_text, placeholder="e.g. Why do clouds form halo rings around the sun?")

col1, col2 = st.columns([1, 5])
with col1:
    search_btn = st.button("🚀 Analyze Concept", type="primary", use_container_width=True)

if (search_btn or default_text) and query_input.strip():
    with st.spinner("🧠 Performing Neural Cross-Attention & Semantic Inference..."):
        t0 = time.time()
        result = engine.explain(query_input.strip(), rerank_top_k=3)
        latency = (time.time() - t0) * 1000

    if result["status"] == "success":
        top = result["top_concept"]
        
        # Metric indicators
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Neural Confidence", top["confidence_level"])
        with m2:
            st.metric("Cosine Similarity", f"{top['cosine_similarity']:.4f}")
        with m3:
            st.metric("Inference Latency", f"{latency:.1f} ms")

        st.markdown(f"""
        <div class="concept-card">
            <span class="chapter-badge">Chapter: {top['chapter']}</span>
            <h3 style="margin-top: 0.3rem; color: #FFFFFF;">{top['question']}</h3>
            <div class="answer-box">
                ✅ Correct Answer: ({top['correct_option']}) {top['correct_answer']}
            </div>
            <div class="explanation-box">
                <b>💡 Scientific Explanation:</b><br/>
                {top['scientific_explanation']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Alternative candidate matches
        if result.get("alternative_matches"):
            with st.expander("🔍 View Alternative Semantic Candidates"):
                for alt in result["alternative_matches"]:
                    st.markdown(f"""
                    **[{alt['chapter']}] {alt['question']}**  
                    *Answer:* ({alt['answer_key']}) {alt['answer_text']}  
                    *Neural Confidence:* `{alt['confidence_prob']*100:.1f}%` | *Cosine:* `{alt['cosine_similarity']}`  
                    *Explanation:* {alt['explanation']}
                    ---
                    """)
    else:
        st.warning("No matching concepts found in the knowledge base.")

# Footer
st.markdown("---")
st.caption("AeroBeacon Semantic ML • Powered by SentenceTransformers, PyTorch & IC Joshi Aviation Meteorology Knowledge Base")
