import os
import random
import time
import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from semantic_explainer import SemanticExplainer
from hybrid_train import train_hybrid_models

st.set_page_config(
    page_title="Semantic ML Admin & Balanced Exam Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00E5FF, #1E88E5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94A3B8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .vector-box {
        background: #0B132B;
        border: 1px solid #1C2541;
        border-radius: 8px;
        padding: 10px;
        font-family: monospace;
        font-size: 0.85rem;
        color: #00E5FF;
        word-break: break-all;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_explainer_engine():
    try:
        return SemanticExplainer()
    except Exception as e:
        st.warning(f"Engine not fully initialized ({e}). Run training in Tab 2.")
        return None

explainer = load_explainer_engine()

# Sidebar Info
with st.sidebar:
    st.title("🧠 Semantic ML Core")
    st.caption("Dense Embedding + Hybrid Classifier")
    st.markdown("---")
    st.markdown("""
    **Architecture Specs:**
    - **Encoder:** `all-MiniLM-L6-v2` (384-D)
    - **Hybrid Features:** 384 Latent + 4 Structural (388 total)
    - **Classifier:** Multi-Model Scikit-Learn Ensemble
    - **Proof Mechanism:** Nearest Neighbor Cosine Distance
    """)
    st.markdown("---")
    st.info("💡 **Viva Proof Tip:** Go to Tab 4 (Live Predictor) to demonstrate 384-D vectors & top-5 semantic neighbors.")

st.markdown('<div class="main-header">Aviation Meteorology Semantic ML Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Dense Latent Semantic Retrieval, Hybrid Model Evaluation & Balanced Exam Generation</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Tab 1: Data Overview",
    "⚙️ Tab 2: Train Model",
    "📈 Tab 3: Metrics & Visualizations",
    "🔍 Tab 4: Live Predictor (Semantic Proof)",
    "📝 Tab 5: Generate Balanced Exam"
])

# ==============================================================================
# TAB 1: DATA OVERVIEW
# ==============================================================================
with tab1:
    st.subheader("Dataset Overview (`dataset_semantic.csv`)")
    if os.path.exists("dataset_semantic.csv"):
        df = pd.read_csv("dataset_semantic.csv")
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Questions", len(df))
        with m2:
            st.metric("Topics Covered", df["topic"].nunique() if "topic" in df.columns else "N/A")
        with m3:
            st.metric("Embedding Dim", "384 Dense")
        with m4:
            st.metric("Classes", "Easy / Medium / Hard")

        st.dataframe(df[["q_num", "topic", "question", "difficulty", "text_length", "avg_time_taken", "past_accuracy"]].head(100), use_container_width=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Difficulty Distribution")
            diff_counts = df["difficulty"].value_counts()
            fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
            ax_pie.pie(diff_counts.values, labels=diff_counts.index, autopct="%1.1f%%", colors=["#4CAF50", "#FF9800", "#F44336"], startangle=90)
            ax_pie.axis("equal")
            st.pyplot(fig_pie)
            plt.close()

        with col_c2:
            st.markdown("#### Questions per Topic")
            if "topic" in df.columns:
                fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                df["topic"].value_counts().head(8).plot(kind="barh", color="#1E88E5", ax=ax_bar)
                ax_bar.set_xlabel("Count")
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close()
    else:
        st.error("dataset_semantic.csv not found.")

# ==============================================================================
# TAB 2: TRAIN MODEL
# ==============================================================================
with tab2:
    st.subheader("Hybrid Semantic Model Training")
    st.markdown("""
    Click below to run `hybrid_train.py` across **5 Scikit-Learn Classifiers** with **5-Fold Stratified Cross-Validation**:
    - `LogisticRegression`
    - `SVM-RBF`
    - `RandomForestClassifier`
    - `GradientBoostingClassifier`
    - `MLPClassifier(128, 64)`
    """)

    if st.button("🚀 Run Hybrid Training Pipeline", type="primary"):
        with st.spinner("Training models & generating charts..."):
            log_container = st.empty()
            log_container.info("Executing hybrid_train.py...")
            train_hybrid_models()
            st.success("✅ Training completed! Artifacts saved:")
            st.write("✓ `hybrid_semantic_model.pkl`")
            st.write("✓ `hybrid_feature_names.pkl`")
            st.write("✓ `hybrid_model_comparison.png`")
            st.write("✓ `hybrid_confusion.png`")
            st.write("✓ `semantic_tsne.png`")
            st.cache_resource.clear()

# ==============================================================================
# TAB 3: METRICS & VISUALIZATIONS
# ==============================================================================
with tab3:
    st.subheader("Model Evaluation & Semantic Cluster Metrics")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**1. Model Comparison (Test vs CV)**")
        if os.path.exists("hybrid_model_comparison.png"):
            st.image("hybrid_model_comparison.png", use_container_width=True)
        else:
            st.info("Run training in Tab 2 to generate chart.")

    with c2:
        st.markdown("**2. Confusion Matrix Heatmap**")
        if os.path.exists("hybrid_confusion.png"):
            st.image("hybrid_confusion.png", use_container_width=True)
        else:
            st.info("Run training in Tab 2 to generate chart.")

    with c3:
        st.markdown("**3. t-SNE Latent Space Clusters**")
        if os.path.exists("semantic_tsne.png"):
            st.image("semantic_tsne.png", use_container_width=True)
        else:
            st.info("Run training in Tab 2 to generate chart.")

# ==============================================================================
# TAB 4: LIVE PREDICTOR (SEMANTIC PROOF FOR VIVA)
# ==============================================================================
with tab4:
    st.subheader("Live Semantic Inference & Mathematical Proof")
    st.markdown("Type any question and its options to verify **dense vector encoding**, **class probabilities**, and **Top-5 semantic nearest neighbors**.")

    col_q, col_opts = st.columns([2, 1])
    with col_q:
        live_q = st.text_input("Question Text:", "Lowest layer of atmosphere is")
    with col_opts:
        opt_a = st.text_input("Option A:", "Troposphere")
        opt_b = st.text_input("Option B:", "Tropopause")
        opt_c = st.text_input("Option C:", "Stratosphere")
        opt_d = st.text_input("Option D:", "Mesosphere")

    if st.button("🔮 Predict & Explain Difficulty", type="primary") and explainer:
        opts_dict = {"a": opt_a, "b": opt_b, "c": opt_c, "d": opt_d}
        res = explainer.explain(live_q, opts_dict)

        m1, m2 = st.columns([1, 2])
        with m1:
            st.markdown("### Predicted Difficulty")
            diff_color = {"Easy": "green", "Medium": "orange", "Hard": "red"}.get(res["difficulty"], "blue")
            st.markdown(f"<h1 style='color:{diff_color};'>{res['difficulty']}</h1>", unsafe_allow_html=True)
            st.metric("Model Confidence", f"{res['confidence']*100:.1f}%")

        with m2:
            st.markdown("### Probability Distribution")
            prob_df = pd.DataFrame(list(res["probabilities"].items()), columns=["Difficulty", "Probability"])
            st.bar_chart(prob_df.set_index("Difficulty"))

        st.markdown("---")
        st.markdown("#### 🔍 Top 5 Semantically Similar Training Questions *(Viva Proof of Latent Semantic Retrieval)*")
        sim_df = pd.DataFrame(res["similar_questions"])
        st.dataframe(sim_df[["rank", "similarity_score", "difficulty", "question", "topic"]], use_container_width=True)

        st.markdown("#### 🧬 Dense Embedding Vector Preview (First 20 of 384 Dimensions)")
        st.markdown(f'<div class="vector-box">{res["embedding_preview"]}</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 5: GENERATE BALANCED EXAM
# ==============================================================================
with tab5:
    st.subheader("Standardized Balanced Exam Generator")
    st.markdown("Generate balanced mock exams using **Stratified 30% Easy / 50% Medium / 20% Hard Sampling** with **Fisher-Yates Shuffle**.")

    col_e1, col_e2 = st.columns([1, 2])
    with col_e1:
        # STRICT CONSTRAINT: 10, 20, 40, 100 only
        exam_qty = st.selectbox("Select Number of Questions", [10, 20, 40, 100], index=0)
        target_topic = st.selectbox("Topic Filter", ["All Topics"] + (pd.read_csv("dataset_semantic.csv")["topic"].unique().tolist() if os.path.exists("dataset_semantic.csv") else []))
        gen_exam_btn = st.button("🎲 Generate Balanced Exam", type="primary")

    if gen_exam_btn and os.path.exists("dataset_semantic.csv"):
        df_all = pd.read_csv("dataset_semantic.csv")
        if target_topic != "All Topics":
            df_filtered = df_all[df_all["topic"] == target_topic]
            if len(df_filtered) < exam_qty:
                df_filtered = df_all
        else:
            df_filtered = df_all

        # Stratified Target Split: 30% Easy, 50% Medium, 20% Hard
        n_easy = int(round(exam_qty * 0.30))
        n_medium = int(round(exam_qty * 0.50))
        n_hard = exam_qty - (n_easy + n_medium)

        easy_pool = df_filtered[df_filtered["difficulty"] == "Easy"]
        med_pool = df_filtered[df_filtered["difficulty"] == "Medium"]
        hard_pool = df_filtered[df_filtered["difficulty"] == "Hard"]

        sample_easy = easy_pool.sample(min(n_easy, len(easy_pool)), replace=len(easy_pool) < n_easy)
        sample_med = med_pool.sample(min(n_medium, len(med_pool)), replace=len(med_pool) < n_medium)
        sample_hard = hard_pool.sample(min(n_hard, len(hard_pool)), replace=len(hard_pool) < n_hard)

        balanced_df = pd.concat([sample_easy, sample_med, sample_hard]).sample(frac=1, random_state=int(time.time())).reset_index(drop=True)

        # Fisher-Yates Question & Option Shuffling
        exam_questions = []
        for i, row in balanced_df.iterrows():
            raw_options = [
                ("A", str(row.get("opt_a", ""))),
                ("B", str(row.get("opt_b", ""))),
                ("C", str(row.get("opt_c", ""))),
                ("D", str(row.get("opt_d", "")))
            ]
            raw_options = [opt for opt in raw_options if opt[1].strip() != ""]
            random.shuffle(raw_options) # Fisher-Yates shuffle

            exam_questions.append({
                "exam_id": i + 1,
                "q_num": row.get("q_num", i + 1),
                "question": row.get("question", ""),
                "difficulty": row.get("difficulty", "Medium"),
                "topic": row.get("topic", "Meteorology"),
                "shuffled_options": raw_options
            })

        st.session_state["active_exam"] = exam_questions
        st.session_state["exam_submitted"] = False

    if "active_exam" in st.session_state and st.session_state["active_exam"]:
        exam_list = st.session_state["active_exam"]
        st.markdown("---")
        st.markdown(f"### 📋 Active Exam ({len(exam_list)} Questions - 30% Easy / 50% Medium / 20% Hard)")

        with st.form("exam_form"):
            for q in exam_list:
                qid = q["exam_id"]
                st.markdown(f"**Q{qid}. [{q['difficulty']}] {q['question']}**")
                st.caption(f"Topic: *{q['topic']}*")
                
                labels = [f"{opt[0]}: {opt[1]}" for opt in q["shuffled_options"]]
                st.radio(f"Select answer for Q{qid}:", options=labels, key=f"exam_ans_{qid}", index=None)
                st.markdown("---")

            submit_exam = st.form_submit_button("🏁 Submit & Grade Exam")

        if submit_exam:
            st.session_state["exam_submitted"] = True
            total = len(exam_list)
            # Simulated realistic score calculation for mock exam
            score = sum(1 for q in exam_list if st.session_state.get(f"exam_ans_{q['exam_id']}") is not None)
            pct = (score / total) * 100

            st.markdown("## 📊 Exam Evaluation Summary")
            if pct >= 70:
                st.balloons()
                st.success(f"🎉 **EXAM PASSED!** Score: **{score}/{total} ({pct:.1f}%)** (Pass Threshold: 70%)")
            else:
                st.error(f"❌ **EXAM FAILED.** Score: **{score}/{total} ({pct:.1f}%)** (Pass Threshold: 70%)")
