/**
 * app.js — AeroBeacon Question Balancing Engine Frontend Logic
 * =============================================================
 * Handles interactive tabs, exam generation, grading, live ML inference,
 * question bank browsing, and real-time backend/ML health metrics.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ── Global State ──────────────────────────────────────────────────────────
  let currentExam = null;
  let allQuestions = [];
  let userAnswers = {};

  // ── Preset Questions for Playground ───────────────────────────────────────
  const PRESET_QUESTIONS = {
    q1: {
      text: "Lowest layer of atmosphere is",
      options: ["Troposphere", "Tropopause", "Stratosphere", "Mesosphere"],
      text_length: 31,
      num_options: 4,
      avg_time_taken: 22,
      past_accuracy: 0.94,
    },
    q2: {
      text: "QNH of an aerodrome 160 m AMSL is 1005 hPa. What is the QFE? (Assume 1 hPa = 8 m)",
      options: ["1010 hPa", "985 hPa", "1005 hPa", "990 hPa"],
      text_length: 83,
      num_options: 4,
      avg_time_taken: 105,
      past_accuracy: 0.28,
    },
    q3: {
      text: "Diurnal variation of surface temperature is greatest when the wind is",
      options: ["Calm", "Light breeze", "Strong", "Gale force"],
      text_length: 69,
      num_options: 4,
      avg_time_taken: 35,
      past_accuracy: 0.88,
    },
    q4: {
      text: "Buys Ballot's Law states that in the Northern Hemisphere, an observer with his back to the wind has",
      options: [
        "High pressure to his left",
        "Low pressure to his left and High pressure to his right",
        "Low pressure directly ahead",
        "High pressure behind",
      ],
      text_length: 99,
      num_options: 4,
      avg_time_taken: 52,
      past_accuracy: 0.65,
    },
    q5: {
      text: "Clear ice (Glaze ice) forms on an aircraft frame in clouds primarily through the impact of",
      options: [
        "Small supercooled droplets that freeze instantly without spreading",
        "Large supercooled water drops that spread backwards before freezing",
        "Dry ice crystals sublimating",
        "Warm rain drops",
      ],
      text_length: 90,
      num_options: 4,
      avg_time_taken: 80,
      past_accuracy: 0.35,
    },
    q6: {
      text: "The Saturated Adiabatic Lapse Rate (SALR) approaches DALR at which temperature level?",
      options: ["at 0°C", "at -15°C", "at -40°C", "at +15°C"],
      text_length: 83,
      num_options: 4,
      avg_time_taken: 60,
      past_accuracy: 0.58,
    },
  };

  // ── Tab Switching ─────────────────────────────────────────────────────────
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");

      tabButtons.forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      tabContents.forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) targetPanel.classList.add("active");
    });
  });

  // ── Ratio Previews Update ─────────────────────────────────────────────────
  const selectExamSize = document.getElementById("select-exam-size");
  function updateRatioPreviews() {
    const size = parseInt(selectExamSize.value, 10) || 20;
    const easyCount = Math.round(size * 0.3);
    const medCount = Math.round(size * 0.5);
    const hardCount = size - easyCount - medCount;

    document.getElementById("target-easy-count").textContent = `${easyCount} Qs`;
    document.getElementById("target-med-count").textContent = `${medCount} Qs`;
    document.getElementById("target-hard-count").textContent = `${hardCount} Qs`;
  }
  selectExamSize.addEventListener("change", updateRatioPreviews);
  updateRatioPreviews();

  // ── Health & Stats Initialization ─────────────────────────────────────────
  async function fetchEngineStats() {
    try {
      const res = await fetch("/api/stats");
      if (!res.ok) throw new Error("Stats fetch failed");
      const data = await res.json();

      // Navbar indicators
      const statusPill = document.getElementById("ml-service-status");
      const pulseDot = document.getElementById("ml-pulse-dot");
      const totalPoolPill = document.getElementById("total-pool-count");
      const modelAccuracyVal = document.getElementById("model-accuracy-val");

      if (data.mlService && data.mlService.available) {
        statusPill.textContent = "ONLINE (RandomForest)";
        pulseDot.style.background = "#00f59b";
        pulseDot.style.boxShadow = "0 0 8px #00f59b";
      } else {
        statusPill.textContent = "STANDBY (Rule Fallback)";
        pulseDot.style.background = "#ffb703";
        pulseDot.style.boxShadow = "0 0 8px #ffb703";
      }

      totalPoolPill.textContent = `${data.totalQuestions || 600} Qs`;
      if (data.mlService?.modelInfo?.model_accuracy) {
        modelAccuracyVal.textContent = `${Math.round(data.mlService.modelInfo.model_accuracy * 100)}%`;
      }

      // Bank tab stats
      document.getElementById("bank-stat-total").textContent = data.totalQuestions || 600;
      document.getElementById("bank-stat-joshi").textContent = data.icJoshiMeteorologyQuestions || 125;
      document.getElementById("bank-stat-easy").textContent = data.difficultyBreakdown?.Easy || 180;
      document.getElementById("bank-stat-med").textContent = data.difficultyBreakdown?.Medium || 300;
      document.getElementById("bank-stat-hard").textContent = data.difficultyBreakdown?.Hard || 120;
    } catch (err) {
      console.warn("Could not fetch engine stats:", err.message);
    }
  }
  fetchEngineStats();

  // ── Exam Generator ────────────────────────────────────────────────────────
  const btnGenerateExam = document.getElementById("btn-generate-exam");
  const examQuestionsList = document.getElementById("exam-questions-list");
  const generationSummary = document.getElementById("generation-summary");
  const btnSubmitExam = document.getElementById("btn-submit-exam");
  const btnExportExam = document.getElementById("btn-export-exam");
  const scoreBanner = document.getElementById("score-banner");

  btnGenerateExam.addEventListener("click", async () => {
    const studentId = document.getElementById("input-student-id").value.trim() || "pilot-cpl-789";
    const totalQuestions = parseInt(selectExamSize.value, 10) || 20;
    const topic = document.getElementById("select-exam-topic").value;

    btnGenerateExam.disabled = true;
    btnGenerateExam.innerHTML = `
      <svg class="btn-icon spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 11-6.219-8.56"/></svg>
      Classifying &amp; Stratifying Exam...
    `;

    try {
      const res = await fetch("/api/exam/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ studentId, totalQuestions, topic: topic || undefined }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || "Failed to generate exam");
      }

      currentExam = await res.json();
      userAnswers = {};
      scoreBanner.style.display = "none";

      renderExamPaper(currentExam);
      renderGenerationAudit(currentExam);
    } catch (err) {
      alert(`Error generating exam: ${err.message}`);
    } finally {
      btnGenerateExam.disabled = false;
      btnGenerateExam.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="btn-icon"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        Generate Balanced Exam
      `;
    }
  });

  function renderGenerationAudit(exam) {
    generationSummary.style.display = "block";
    const meta = exam.metadata || {};
    const breakdown = meta.difficultyBreakdown || { easy: 0, medium: 0, hard: 0 };
    const total = exam.questions.length;

    document.getElementById("gen-actual-count").textContent = total;
    document.getElementById("gen-easy-stat").textContent = `${breakdown.easy} (${Math.round((breakdown.easy / total) * 100)}%)`;
    document.getElementById("gen-med-stat").textContent = `${breakdown.medium} (${Math.round((breakdown.medium / total) * 100)}%)`;
    document.getElementById("gen-hard-stat").textContent = `${breakdown.hard} (${Math.round((breakdown.hard / total) * 100)}%)`;

    const sourceBadge = document.getElementById("gen-source-badge");
    if (meta.predictionSource === "ml_service") {
      sourceBadge.textContent = "LOCAL ML ENGINE (RandomForest)";
      sourceBadge.style.color = "#00f59b";
      sourceBadge.style.borderColor = "rgba(0, 245, 155, 0.4)";
    } else {
      sourceBadge.textContent = "RULE FALLBACK";
      sourceBadge.style.color = "#ffb703";
      sourceBadge.style.borderColor = "rgba(255, 183, 3, 0.4)";
    }

    // Set proportional distribution bar
    document.getElementById("bar-easy").style.width = `${(breakdown.easy / total) * 100}%`;
    document.getElementById("bar-med").style.width = `${(breakdown.medium / total) * 100}%`;
    document.getElementById("bar-hard").style.width = `${(breakdown.hard / total) * 100}%`;
  }

  function renderExamPaper(exam) {
    examQuestionsList.innerHTML = "";
    btnSubmitExam.style.display = "inline-block";
    btnExportExam.style.display = "inline-block";

    document.getElementById("exam-paper-title").textContent = `Mock Examination (${exam.questions.length} Questions)`;

    exam.questions.forEach((q, index) => {
      const qCard = document.createElement("div");
      qCard.className = "question-card";
      qCard.id = `q-card-${index}`;

      const diffClass =
        q.predictedDifficulty === "Easy"
          ? "tag-easy"
          : q.predictedDifficulty === "Hard"
          ? "tag-hard"
          : "tag-medium";

      const confidencePct = Math.round((q.predictionConfidence || 1.0) * 100);

      const optionsHtml = (q.options || [])
        .map((opt, optIdx) => {
          const letter = String.fromCharCode(65 + optIdx);
          return `
          <label class="q-option-label" id="opt-label-${index}-${optIdx}">
            <input type="radio" name="question_${index}" value="${optIdx}" class="opt-radio">
            <span class="opt-letter">${letter}.</span>
            <span class="opt-text">${opt}</span>
          </label>
        `;
        })
        .join("");

      qCard.innerHTML = `
        <div class="q-header">
          <span class="q-number">QUESTION ${index + 1} OF ${exam.questions.length}</span>
          <div class="q-meta">
            <span class="q-topic-tag">${q.topic || "Meteorology"}</span>
            <span class="diff-tag ${diffClass}">${q.predictedDifficulty || "Medium"} (${confidencePct}%)</span>
          </div>
        </div>
        <div class="q-text">${q.text}</div>
        <div class="q-options">${optionsHtml}</div>
        <div class="q-explanation" style="display: none;" id="exp-${index}">
          <strong>Answer Key:</strong> Option ${String.fromCharCode(65 + q.correctIndex)} &bull; ${q.explanation || "Official DGCA Syllabus Concept"}
        </div>
      `;

      examQuestionsList.appendChild(qCard);

      // Track selection
      const radios = qCard.querySelectorAll('input[type="radio"]');
      radios.forEach((r) => {
        r.addEventListener("change", (e) => {
          userAnswers[index] = parseInt(e.target.value, 10);
        });
      });
    });
  }

  // ── Exam Submission & Grading ─────────────────────────────────────────────
  btnSubmitExam.addEventListener("click", () => {
    if (!currentExam) return;

    let correctCount = 0;
    const total = currentExam.questions.length;

    currentExam.questions.forEach((q, index) => {
      const selected = userAnswers[index];
      const correct = q.correctIndex;
      const expBox = document.getElementById(`exp-${index}`);
      if (expBox) expBox.style.display = "block";

      const correctLabel = document.getElementById(`opt-label-${index}-${correct}`);
      if (correctLabel) correctLabel.classList.add("correct");

      if (selected !== undefined) {
        if (selected === correct) {
          correctCount++;
        } else {
          const wrongLabel = document.getElementById(`opt-label-${index}-${selected}`);
          if (wrongLabel) wrongLabel.classList.add("incorrect");
        }
      }
    });

    const pct = Math.round((correctCount / total) * 100);
    const passed = pct >= 70; // Standard DGCA passing percentage is 70%

    scoreBanner.style.display = "block";
    document.getElementById("score-percentage").textContent = `${pct}%`;
    document.getElementById("score-summary").textContent = `You answered ${correctCount} of ${total} questions correctly.`;

    const gradeTitle = document.getElementById("score-grade");
    const circle = document.getElementById("score-circle");

    if (passed) {
      gradeTitle.textContent = "PASS • DGCA Regulatory Standard Met (≥70%)";
      gradeTitle.style.color = "#00f59b";
      circle.style.borderColor = "#00f59b";
      circle.style.color = "#00f59b";
    } else {
      gradeTitle.textContent = "RE-ATTEMPT RECOMMENDED • DGCA Pass Mark is 70%";
      gradeTitle.style.color = "#ffb703";
      circle.style.borderColor = "#ffb703";
      circle.style.color = "#ffb703";
    }

    scoreBanner.scrollIntoView({ behavior: "smooth" });
  });

  // ── Export Exam ───────────────────────────────────────────────────────────
  btnExportExam.addEventListener("click", () => {
    if (!currentExam) return;
    const blob = new Blob([JSON.stringify(currentExam, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `aerobeacon_balanced_exam_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // ── ML Classifier Lab ─────────────────────────────────────────────────────
  const sliderTextLen = document.getElementById("play-text-length");
  const sliderNumOpts = document.getElementById("play-num-options");
  const sliderAvgTime = document.getElementById("play-avg-time");
  const sliderPastAcc = document.getElementById("play-past-accuracy");

  const lblTextLen = document.getElementById("lbl-text-length");
  const lblNumOpts = document.getElementById("lbl-num-options");
  const lblAvgTime = document.getElementById("lbl-avg-time");
  const lblPastAcc = document.getElementById("lbl-past-accuracy");

  const qText = document.getElementById("play-question-text");
  const previewJson = document.getElementById("play-json-preview");
  const selectPreset = document.getElementById("select-preset-question");
  const btnRunPrediction = document.getElementById("btn-run-prediction");

  function updatePlaygroundPreview() {
    const payload = {
      text_length: parseInt(sliderTextLen.value, 10),
      num_options: parseInt(sliderNumOpts.value, 10),
      avg_time_taken: parseInt(sliderAvgTime.value, 10),
      past_accuracy: parseFloat((parseInt(sliderPastAcc.value, 10) / 100).toFixed(2)),
    };

    lblTextLen.textContent = payload.text_length;
    lblNumOpts.textContent = payload.num_options;
    lblAvgTime.textContent = payload.avg_time_taken;
    lblPastAcc.textContent = `${sliderPastAcc.value}%`;

    previewJson.textContent = JSON.stringify(payload, null, 2);
  }

  [sliderTextLen, sliderNumOpts, sliderAvgTime, sliderPastAcc].forEach((s) =>
    s.addEventListener("input", updatePlaygroundPreview)
  );

  qText.addEventListener("input", () => {
    sliderTextLen.value = Math.min(Math.max(qText.value.length, 20), 400);
    updatePlaygroundPreview();
  });

  selectPreset.addEventListener("change", () => {
    const preset = PRESET_QUESTIONS[selectPreset.value];
    if (preset) {
      qText.value = preset.text;
      sliderTextLen.value = preset.text_length;
      sliderNumOpts.value = preset.num_options;
      sliderAvgTime.value = preset.avg_time_taken;
      sliderPastAcc.value = Math.round(preset.past_accuracy * 100);
      updatePlaygroundPreview();
      runPrediction();
    }
  });

  async function runPrediction() {
    const payload = {
      text: qText.value,
      options: ["A", "B", "C", "D"].slice(0, parseInt(sliderNumOpts.value, 10)),
      avgTimeTaken: parseInt(sliderAvgTime.value, 10),
      pastAccuracy: parseFloat((parseInt(sliderPastAcc.value, 10) / 100).toFixed(2)),
    };

    btnRunPrediction.disabled = true;

    try {
      const res = await fetch("/api/questions/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error("Inference call failed");
      const data = await res.json();

      const diff = data.difficulty || "Medium";
      const conf = Math.round((data.confidence || 1.0) * 100);
      const probs = data.probabilities || { Easy: 0, Medium: 0, Hard: 0 };

      // Update UI hero box
      const heroTitle = document.getElementById("res-difficulty-title");
      heroTitle.textContent = diff;
      heroTitle.className = `hero-difficulty diff-${diff.toLowerCase()}`;

      document.getElementById("res-confidence-badge").textContent = `Confidence: ${conf}%`;

      const expText = document.getElementById("res-explanation-text");
      if (diff === "Easy") {
        expText.textContent = `High historical accuracy (${sliderPastAcc.value}%) and rapid response time (${sliderAvgTime.value}s) classify this question as foundational recall.`;
      } else if (diff === "Hard") {
        expText.textContent = `High solving time (${sliderAvgTime.value}s) and low cohort accuracy (${sliderPastAcc.value}%) indicate complex calculation or multi-step reasoning.`;
      } else {
        expText.textContent = `Balanced solving time (${sliderAvgTime.value}s) and standard accuracy (${sliderPastAcc.value}%) match DGCA core conceptual benchmarks.`;
      }

      // Update probability bars
      const pEasy = ((probs.Easy || 0) * 100).toFixed(1);
      const pMed = ((probs.Medium || 0) * 100).toFixed(1);
      const pHard = ((probs.Hard || 0) * 100).toFixed(1);

      document.getElementById("prob-easy-bar").style.width = `${pEasy}%`;
      document.getElementById("prob-easy-pct").textContent = `${pEasy}%`;

      document.getElementById("prob-med-bar").style.width = `${pMed}%`;
      document.getElementById("prob-med-pct").textContent = `${pMed}%`;

      document.getElementById("prob-hard-bar").style.width = `${pHard}%`;
      document.getElementById("prob-hard-pct").textContent = `${pHard}%`;
    } catch (err) {
      console.error("Prediction error:", err);
    } finally {
      btnRunPrediction.disabled = false;
    }
  }

  btnRunPrediction.addEventListener("click", runPrediction);
  updatePlaygroundPreview();

  // ── Question Bank Explorer ────────────────────────────────────────────────
  const questionsTableBody = document.getElementById("questions-table-body");
  const bankSearchInput = document.getElementById("bank-search-input");
  const bankTopicFilter = document.getElementById("bank-topic-filter");
  const bankDiffFilter = document.getElementById("bank-diff-filter");

  async function loadQuestionsBank() {
    try {
      const res = await fetch("/api/questions?limit=1000");
      if (!res.ok) throw new Error("Could not load questions");
      const data = await res.json();
      allQuestions = data.questions || [];
      renderQuestionBankTable();
    } catch (err) {
      console.warn("Could not load bank table:", err.message);
    }
  }

  function renderQuestionBankTable() {
    const search = bankSearchInput.value.toLowerCase();
    const topic = bankTopicFilter.value;
    const diff = bankDiffFilter.value;

    const filtered = allQuestions.filter((q) => {
      const matchSearch = !search || q.text.toLowerCase().includes(search) || (q.subtopic && q.subtopic.toLowerCase().includes(search));
      const matchTopic = !topic || q.topic === topic;
      const matchDiff = !diff || q.difficulty === diff;
      return matchSearch && matchTopic && matchDiff;
    });

    questionsTableBody.innerHTML = "";

    if (filtered.length === 0) {
      questionsTableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-dim); padding: 2rem;">No questions match current filters.</td></tr>`;
      return;
    }

    filtered.slice(0, 150).forEach((q, idx) => {
      const tr = document.createElement("tr");
      const diffTag =
        q.difficulty === "Easy"
          ? '<span class="diff-tag tag-easy">Easy</span>'
          : q.difficulty === "Hard"
          ? '<span class="diff-tag tag-hard">Hard</span>'
          : '<span class="diff-tag tag-medium">Medium</span>';

      const isJoshi = (q.id && q.id.startsWith("joshi")) || q.topic;
      const joshiBadge = isJoshi ? `<span style="font-size: 0.65rem; color: #00e5ff; font-weight: 700;">★ IC JOSHI METEOROLOGY</span>` : "";

      tr.innerHTML = `
        <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--text-dim);">${q.id || `Q-${idx+1}`}</td>
        <td>
          <div style="font-weight: 500; margin-bottom: 0.2rem;">${q.text}</div>
          ${joshiBadge}
        </td>
        <td>
          <div style="font-size: 0.8rem; font-weight: 600;">${q.topic}</div>
          <div style="font-size: 0.72rem; color: var(--text-muted);">${q.subtopic || ""}</div>
        </td>
        <td>${diffTag}</td>
        <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;">${q.avgTimeTaken}s</td>
        <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;">${Math.round((q.pastAccuracy || 0.5) * 100)}%</td>
      `;
      questionsTableBody.appendChild(tr);
    });
  }

  bankSearchInput.addEventListener("input", renderQuestionBankTable);
  bankTopicFilter.addEventListener("change", renderQuestionBankTable);
  bankDiffFilter.addEventListener("change", renderQuestionBankTable);

  loadQuestionsBank();
});
