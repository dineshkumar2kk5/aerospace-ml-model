"""
semantic_admin_ui.py — Streamlit Administrative Dashboard for Semantic ML (Port 8502)
=====================================================================================

Interactive web dashboard for visualizing the AeroBeacon semantic ML upgrade:
  - Tab 1: Dataset exploration (806 IC Joshi aviation meteorology questions)
  - Tab 2: Subprocess-driven model training with live streaming stdout/stderr
  - Tab 3: Model benchmarks, confusion matrix, and 2D t-SNE semantic manifold
  - Tab 4: Real-time inference with top-5 cosine similarity explanation and vector preview
  - Tab 5: Dynamic balanced exam generator (30% Easy, 50% Medium, 20% Hard) with interactive testing

Usage:
  streamlit run semantic_admin_ui.py --server.port 8502
"""

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AeroBeacon — Semantic ML Engine",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .stat-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .viva-badge {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FLASK_API_URL = "http://127.0.0.1:5002"


def resolve_file(filename: str) -> Path:
    """
    Locate file across workspace candidate paths.
    """
    candidates = [
        Path(filename),
        Path(__file__).resolve().parent / filename,
        Path(__file__).resolve().parent / "ml" / filename,
        Path(__file__).resolve().parent.parent / "ml" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return Path(filename).resolve()


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """
    Load dataset_semantic.csv into cached pandas DataFrame.
    """
    csv_path = resolve_file("dataset_semantic.csv")
    if not csv_path.exists():
        st.error(f"❌ Could not find {csv_path.name}. Run semantic_embedder.py first!")
        return pd.DataFrame()
    return pd.read_csv(csv_path)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">✈️ AeroBeacon — Semantic ML Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Aviation Meteorology Question Balancing Engine & Dense Vector Explainer (Port 8502)</div>',
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/airplane-take-off.png", width=70)
    st.title("AeroBeacon Admin")
    st.markdown("**Platform Status**")

    # Check Microservice Connectivity
    try:
        r = requests.get(f"{FLASK_API_URL}/health", timeout=1.5)
        if r.status_code == 200:
            health_info = r.json()
            st.success(f"ML Service Online (Port 5002)\n- Model: {health_info.get('model_type')}")
        else:
            st.warning("ML Service returned non-200 status.")
    except Exception:
        st.error("ML Service Offline\nRun `python semantic_ml_service.py` to start.")

    st.markdown("---")
    st.markdown(
        """
        **System Specs:**
        - **Embedding**: all-MiniLM-L6-v2 (384d)
        - **Total Features**: 388 (Hybrid)
        - **Question Bank**: 806 IC Joshi MCQs
        - **DGCA Syllabus**: 28 Topics
        - **Passing Grade**: ≥ 70%
        """
    )
    st.markdown("---")
    st.caption("AeroBeacon v2.0 • Hybrid Semantic Upgrade")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Semantic Data Overview",
    "🧠 Train Hybrid Model",
    "📈 Metrics Dashboard",
    "🔍 Live Semantic Predictor",
    "🎯 Generate Balanced Exam",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Semantic Data Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("DGCA Aviation Meteorology Question Corpus")
    df = load_dataset()

    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Questions", len(df))
        col2.metric("Syllabus Topics", df["topic"].nunique() if "topic" in df.columns else 28)
        easy_cnt = (df["difficulty"] == "Easy").sum() if "difficulty" in df.columns else 0
        med_cnt = (df["difficulty"] == "Medium").sum() if "difficulty" in df.columns else 0
        hard_cnt = (df["difficulty"] == "Hard").sum() if "difficulty" in df.columns else 0
        col3.metric("Difficulty Split", f"{easy_cnt}E / {med_cnt}M / {hard_cnt}H")
        col4.metric("Avg Question Length", f"{int(df['question'].str.len().mean())} chars")

        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("##### 📊 Difficulty Distribution")
            diff_counts = df["difficulty"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#10B981", "#F59E0B", "#EF4444"]
            diff_counts.plot(kind="bar", ax=ax, color=["#10B981", "#3B82F6", "#EF4444"][:len(diff_counts)])
            ax.set_ylabel("Count")
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with chart_col2:
            st.markdown("##### 🥧 Top 8 Topics Distribution")
            if "topic" in df.columns:
                top_topics = df["topic"].value_counts().head(8)
                fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
                ax_pie.pie(top_topics, labels=top_topics.index, autopct="%1.1f%%", startangle=140, colors=plt.cm.Paired.colors)
                ax_pie.axis("equal")
                plt.tight_layout()
                st.pyplot(fig_pie)
                plt.close()

        st.markdown("---")
        st.markdown("##### 📋 Question Bank Preview (First 20 Questions)")
        preview_cols = [c for c in ["q_num", "topic", "question", "difficulty", "avg_time_taken"] if c in df.columns]
        st.dataframe(df[preview_cols].head(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Train Hybrid Model
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Train Hybrid Semantic-Structural Classifier")
    st.markdown(
        """
        Trigger the full training pipeline (`hybrid_train.py`) to benchmark **5 candidate algorithms**:
        `LogisticRegression`, `SVM (RBF)`, `RandomForest`, `GradientBoosting`, and `MLP Neural Network`.
        """
    )

    if st.button("🚀 Start Hybrid Training Pipeline", type="primary"):
        train_script = resolve_file("hybrid_train.py")
        if not train_script.exists():
            st.error(f"❌ Script not found: {train_script.name}")
        else:
            status_box = st.empty()
            log_container = st.empty()
            status_box.info("⚙️ Training in progress... Streaming standard output:")

            try:
                cmd = [sys.executable, str(train_script)]
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(train_script.parent),
                )

                log_lines = []
                for line in iter(process.stdout.readline, ""):
                    log_lines.append(line)
                    log_container.code("".join(log_lines[-25:]), language="bash")

                process.stdout.close()
                return_code = process.wait()

                if return_code == 0:
                    status_box.success("✅ Model trained successfully! Artifacts updated.")
                    st.balloons()
                    # Trigger hot reload on Flask API if running
                    try:
                        requests.post(f"{FLASK_API_URL}/retrain-hook", timeout=2)
                        st.info("📡 Flask ML service hot-reloaded the new model in memory.")
                    except Exception:
                        pass
                else:
                    status_box.error(f"❌ Training failed with exit code {return_code}.")
            except Exception as ex:
                status_box.error(f"❌ Execution error: {ex}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Metrics Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Evaluation & Semantic Geometry")

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("##### 🏆 5-Fold Stratified Cross-Validation Benchmark")
        comp_img = resolve_file("hybrid_model_comparison.png")
        if comp_img.exists():
            st.image(str(comp_img), caption="Test Accuracy vs 5-Fold CV (Higher is Better)", use_container_width=True)
        else:
            st.info("Comparison chart not found. Run training in Tab 2 to generate.")

    with img_col2:
        st.markdown("##### 🎯 Confusion Matrix (Best Hybrid Model)")
        conf_img = resolve_file("hybrid_confusion.png")
        if conf_img.exists():
            st.image(str(conf_img), caption="Normalized Confusion Heatmap across Easy/Medium/Hard", use_container_width=True)
        else:
            st.info("Confusion matrix not found. Run training in Tab 2 to generate.")

    st.markdown("---")
    st.markdown("##### 🌌 2D t-SNE Semantic Manifold")
    st.markdown(
        """
        <span class="viva-badge">VIVA PROOF</span>
        *Demonstrates how 384-dimensional sentence embeddings map questions into distinct, 
        coherent topological neighborhoods based on physical concepts and mathematical complexity.*
        """,
        unsafe_allow_html=True,
    )
    tsne_img = resolve_file("semantic_tsne.png")
    if tsne_img.exists():
        st.image(str(tsne_img), caption="Semantic clusters prove the model learned meaning", use_container_width=True)
    else:
        st.info("t-SNE chart not found. Run training in Tab 2 to generate.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Live Semantic Predictor
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Real-Time Semantic Difficulty Predictor & KNN Explainer")
    st.markdown(
        """
        Input any DGCA question. The system will encode it into dense semantic vector space, 
        predict difficulty tier, and retrieve the **top-5 most semantically aligned reference questions**.
        """
    )

    pred_col1, pred_col2 = st.columns([3, 2])

    with pred_col1:
        sample_q = "Calculate ISA deviation at FL190 with temperature -60C"
        user_question = st.text_area("Question Text", value=sample_q, height=100)
        user_options = st.text_input("Options (comma separated)", value="-22°C, -37°C, -15°C, Normal")
        predict_btn = st.button("🔮 Predict Difficulty & Explain", type="primary")

    if predict_btn and user_question.strip():
        opts_list = [o.strip() for o in user_options.split(",") if o.strip()]
        payload = {"question": user_question.strip(), "options": opts_list}

        try:
            with st.spinner("🧠 Querying Semantic ML Service..."):
                resp = requests.post(f"{FLASK_API_URL}/predict", json=payload, timeout=5)

            if resp.status_code == 200:
                result = resp.json()
                diff = result["difficulty"]
                conf = result["confidence"]
                probs = result["probabilities"]
                similars = result["similar_questions"]

                with pred_col2:
                    st.markdown("##### 🎯 Prediction Output")
                    color = "#10B981" if diff == "Easy" else ("#F59E0B" if diff == "Medium" else "#EF4444")
                    st.markdown(
                        f"""
                        <div style="background-color: {color}20; border: 2px solid {color}; border-radius: 8px; padding: 20px; text-align: center;">
                            <h4 style="margin:0; color: #334155;">PREDICTED DIFFICULTY</h4>
                            <h1 style="margin:5px 0; color: {color};">{diff.upper()}</h1>
                            <p style="margin:0; font-weight: bold; color: #475569;">Confidence: {conf*100:.1f}%</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("---")
                chart_c, vec_c = st.columns([1, 1])

                with chart_c:
                    st.markdown("##### 📊 Class Probability Distribution")
                    prob_df = pd.DataFrame(list(probs.items()), columns=["Tier", "Probability"]).set_index("Tier")
                    st.bar_chart(prob_df)

                with vec_c:
                    st.markdown("##### 📐 Dense Embedding Preview (First 20 Dimensions)")
                    try:
                        exp_resp = requests.post(f"{FLASK_API_URL}/explain", json=payload, timeout=3)
                        if exp_resp.status_code == 200:
                            exp_data = exp_resp.json()
                            vec = exp_data.get("embedding_preview", [])
                            st.code("Vector[0:20]:\n" + ", ".join(f"{x:+.4f}" for x in vec), language="python")
                            st.caption(f"Vector L2 Norm: {exp_data.get('embedding_norm', 1.0)}")
                    except Exception:
                        st.caption("Embedding vector preview unavailable.")

                st.markdown("---")
                st.markdown("##### 🔍 Top 5 Semantically Similar Training Questions (Core Viva Proof)")
                sim_df = pd.DataFrame(similars)
                if not sim_df.empty:
                    sim_df = sim_df.rename(columns={
                        "similarity": "Cosine Similarity",
                        "difficulty": "Difficulty",
                        "question": "Reference Question",
                        "topic": "Topic",
                    })
                    st.dataframe(sim_df[["Cosine Similarity", "Difficulty", "Topic", "Reference Question"]], use_container_width=True)
            else:
                st.error(f"API Error ({resp.status_code}): {resp.text}")
        except Exception as e:
            st.error(f"❌ Could not connect to Semantic ML Service at {FLASK_API_URL}. Is it running?")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Generate Balanced Exam
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("DGCA Balanced Exam Generator & Interactive Test Simulator")
    st.markdown(
        """
        Generates standard DGCA mock exams with strictly balanced difficulty quotas:
        - **30% Easy** (definitions, fundamental facts)
        - **50% Medium** (conceptual relationships)
        - **20% Hard** (multi-step calculations, operational judgment)
        """
    )

    exam_col1, exam_col2 = st.columns([1, 2])
    with exam_col1:
        total_q = st.selectbox("Exam Question Count", options=[10, 20, 40, 100], index=1)
        gen_btn = st.button("🎲 Generate Balanced Exam", type="primary")

    df_full = load_dataset()

    if gen_btn and not df_full.empty:
        # Calculate target quota counts
        n_easy = int(round(total_q * 0.30))
        n_hard = int(round(total_q * 0.20))
        n_medium = total_q - n_easy - n_hard

        # Filter candidate pools
        easy_pool = df_full[df_full["difficulty"] == "Easy"].to_dict("records")
        med_pool = df_full[df_full["difficulty"] == "Medium"].to_dict("records")
        hard_pool = df_full[df_full["difficulty"] == "Hard"].to_dict("records")

        selected_easy = random.sample(easy_pool, min(n_easy, len(easy_pool)))
        selected_med = random.sample(med_pool, min(n_medium, len(med_pool)))
        selected_hard = random.sample(hard_pool, min(n_hard, len(hard_pool)))

        exam_questions = selected_easy + selected_med + selected_hard

        # Fisher-Yates shuffle questions
        random.shuffle(exam_questions)

        # Structure questions for testing
        processed_exam = []
        for idx, item in enumerate(exam_questions, 1):
            opts = [str(item.get(c, "")).strip() for c in ["opt_a", "opt_b", "opt_c", "opt_d"] if pd.notna(item.get(c)) and str(item.get(c, "")).strip()]
            if not opts:
                opts = ["Option A", "Option B", "Option C", "Option D"]

            correct_ans = opts[0]  # Reference answer anchor
            # Fisher-Yates shuffle options
            shuffled_opts = list(opts)
            random.shuffle(shuffled_opts)

            processed_exam.append({
                "exam_id": idx,
                "question": item.get("question", f"Question {idx}"),
                "topic": item.get("topic", "Meteorology"),
                "difficulty": item.get("difficulty", "Medium"),
                "options": shuffled_opts,
                "correct_answer": correct_ans,
            })

        st.session_state["active_exam"] = processed_exam
        st.session_state["exam_submitted"] = False
        st.session_state["user_answers"] = {}
        st.success(f"Exam generated with {len(processed_exam)} questions ({n_easy} Easy, {n_medium} Medium, {n_hard} Hard)!")

    # Render Interactive Exam if present in session state
    if "active_exam" in st.session_state and st.session_state["active_exam"]:
        exam = st.session_state["active_exam"]
        st.markdown("---")
        st.markdown(f"#### 📝 Interactive Exam Session ({len(exam)} Questions)")

        # Form for submitting answers
        with st.form("exam_submission_form"):
            user_responses = {}
            for item in exam:
                q_num = item["exam_id"]
                st.markdown(f"**Q{q_num}. {item['question']}**")
                st.caption(f"Topic: {item['topic']} | Tier: {item['difficulty']}")
                user_responses[q_num] = st.radio(
                    label=f"Options for Q{q_num}",
                    options=item["options"],
                    key=f"radio_q_{q_num}",
                    label_visibility="collapsed",
                )
                st.write("")

            submitted = st.form_submit_button("🏁 Submit Exam for Scoring", type="primary")

        if submitted:
            st.session_state["exam_submitted"] = True
            st.session_state["user_answers"] = user_responses

            correct_count = 0
            topic_stats = {}

            for item in exam:
                qid = item["exam_id"]
                user_ans = user_responses.get(qid)
                is_correct = (user_ans == item["correct_answer"])
                if is_correct:
                    correct_count += 1

                t = item["topic"]
                if t not in topic_stats:
                    topic_stats[t] = {"correct": 0, "total": 0}
                topic_stats[t]["total"] += 1
                if is_correct:
                    topic_stats[t]["correct"] += 1

            score_pct = (correct_count / len(exam)) * 100
            passed = score_pct >= 70.0

            st.markdown("---")
            res_box = st.container()
            if passed:
                res_box.success(f"🎉 **RESULT: PASS** — Score: {score_pct:.1f}% ({correct_count}/{len(exam)} correct). DGCA Passing threshold (70%) achieved!")
            else:
                res_box.error(f"❌ **RESULT: FAIL** — Score: {score_pct:.1f}% ({correct_count}/{len(exam)} correct). Minimum 70% required by DGCA.")

            # Topic breakdown table
            st.markdown("##### 📊 Topic-by-Topic Performance Breakdown")
            t_rows = []
            for t, s in topic_stats.items():
                acc = (s["correct"] / s["total"]) * 100
                t_rows.append({"Topic": t, "Correct": s["correct"], "Total": s["total"], "Accuracy": f"{acc:.1f}%"})
            st.dataframe(pd.DataFrame(t_rows), use_container_width=True)

        # JSON Export
        st.markdown("---")
        json_export = json.dumps(exam, indent=2)
        st.download_button(
            label="💾 Export Exam to JSON",
            data=json_export,
            file_name="dgca_balanced_exam.json",
            mime="application/json",
        )
