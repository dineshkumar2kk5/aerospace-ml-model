import streamlit as st
import json
import time
from semantic_explainer import SemanticExplainer
from quiz_generator import QuizGenerator

st.set_page_config(
    page_title="AeroBeacon - Semantic Meteorology AI & Quiz Engine",
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
def get_engines():
    explainer = SemanticExplainer()
    quiz_gen = QuizGenerator()
    return explainer, quiz_gen

with st.spinner("✈️ Loading Aviation Semantic ML Core & Quiz Database..."):
    engine, quiz_engine = get_engines()

# Header
st.markdown('<div class="main-title">AeroBeacon - Aviation Meteorology AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Semantic Retrieval & AI Quiz Generation for DGCA CPL/ATPL Pilots</div>', unsafe_allow_html=True)

# Tabs: Semantic Search vs. Dynamic Quiz
tab_search, tab_quiz = st.tabs(["🔍 Semantic Concept Search", "📝 Interactive Quiz Generator"])

# ----------------- TAB 1: SEMANTIC SEARCH -----------------
with tab_search:
    st.subheader("Neural Latent Query & Reasoning")
    query_input = st.text_input(
        "Ask any meteorology question or concept in natural language:",
        placeholder="e.g. Why does the tropopause height vary with latitude?"
    )

    if st.button("🚀 Analyze Concept", type="primary", use_container_width=False) and query_input.strip():
        with st.spinner("🧠 Performing Neural Cross-Attention Inference..."):
            t0 = time.time()
            result = engine.explain(query_input.strip(), rerank_top_k=3)
            latency = (time.time() - t0) * 1000

        if result["status"] == "success":
            top = result["top_concept"]
            
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

            if result.get("alternative_matches"):
                with st.expander("🔍 View Alternative Semantic Candidates"):
                    for alt in result["alternative_matches"]:
                        st.markdown(f"""
                        **[{alt['chapter']}] {alt['question']}**  
                        *Answer:* ({alt['answer_key']}) {alt['answer_text']}  
                        *Neural Confidence:* `{alt['confidence_prob']*100:.1f}%`  
                        *Explanation:* {alt['explanation']}
                        ---
                        """)
        else:
            st.warning("No matching concepts found in the knowledge base.")

# ----------------- TAB 2: INTERACTIVE QUIZ GENERATOR -----------------
with tab_quiz:
    st.subheader("Aviation Meteorology Mock Exam & Quiz Generator")
    
    col_mode, col_qty, col_chap = st.columns([1.5, 1, 1.5])
    
    with col_mode:
        quiz_mode = st.selectbox(
            "Quiz Mode",
            ["Chapter / Topic Wise", "Full Syllabus Mock Test", "Semantic Concept Adaptive"]
        )
        
    with col_qty:
        num_q = st.slider("Number of Questions", min_value=3, max_value=25, value=5)
        
    with col_chap:
        if quiz_mode == "Chapter / Topic Wise":
            selected_chapter = st.selectbox("Select Chapter", quiz_engine.get_available_chapters())
        elif quiz_mode == "Semantic Concept Adaptive":
            concept_prompt = st.text_input("Target Concept / Topic", value="Thunderstorm downdraughts and microburst")
        else:
            st.info("Random selection across all 659 questions.")

    if st.button("🎲 Generate New Quiz", type="primary"):
        if quiz_mode == "Chapter / Topic Wise":
            st.session_state["current_quiz"] = quiz_engine.generate_random_quiz(num_questions=num_q, chapter=selected_chapter)
        elif quiz_mode == "Semantic Concept Adaptive":
            st.session_state["current_quiz"] = quiz_engine.generate_semantic_quiz(concept_prompt, num_questions=num_q)
        else:
            st.session_state["current_quiz"] = quiz_engine.generate_random_quiz(num_questions=num_q)
        st.session_state["quiz_submitted"] = False
        st.session_state["user_answers"] = {}

    # Render Current Quiz
    if "current_quiz" in st.session_state and st.session_state["current_quiz"]:
        quiz_data = st.session_state["current_quiz"]
        st.markdown("---")
        st.markdown(f"### 📋 Active Quiz ({len(quiz_data)} Questions)")

        with st.form("quiz_form"):
            for item in quiz_data:
                qid = item["quiz_id"]
                st.markdown(f"**Question {qid}:** {item['question']}")
                st.caption(f"Chapter: *{item['chapter']}*")
                
                options_dict = item["options"]
                option_labels = [f"({k}) {v}" for k, v in options_dict.items()]
                
                user_choice = st.radio(
                    f"Select your answer for Q{qid}:",
                    options=option_labels,
                    key=f"q_{qid}",
                    index=None
                )
                st.markdown("---")

            submit_quiz = st.form_submit_button("🏁 Submit & Grade Exam")

        if submit_quiz:
            st.session_state["quiz_submitted"] = True
            score = 0
            total = len(quiz_data)

            st.markdown("## 📊 Quiz Results & Detailed Explanations")
            for item in quiz_data:
                qid = item["quiz_id"]
                correct_k = item["correct_answer"]
                correct_text = item["options"].get(correct_k, "")
                user_selected = st.session_state.get(f"q_{qid}")
                
                user_key = user_selected[1] if user_selected and len(user_selected) > 1 else ""
                is_correct = (user_key.lower() == correct_k.lower())

                if is_correct:
                    score += 1
                    st.success(f"**Q{qid}: Correct!** ✅  \n**Answer:** ({correct_k}) {correct_text}")
                else:
                    st.error(f"**Q{qid}: Incorrect** ❌  \n**Your Answer:** {user_selected or 'None'}  \n**Correct Answer:** ({correct_k}) {correct_text}")

                st.info(f"**💡 Scientific Rationale:** {item['explanation']}")
                st.markdown("---")

            pct = (score / total) * 100
            if pct >= 70:
                st.balloons()
                st.success(f"🎉 **Exam Passed!** Final Score: **{score}/{total} ({pct:.1f}%)**")
            else:
                st.warning(f"⚠️ **Review Needed.** Final Score: **{score}/{total} ({pct:.1f}%)** (Pass mark: 70%)")

# Footer
st.markdown("---")
st.caption("AeroBeacon Semantic ML & Quiz Engine • Powered by PyTorch, SentenceTransformers & IC Joshi Aviation Meteorology Knowledge Base")
