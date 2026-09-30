/**
 * mlQuestionService.js — ML-Based Question Balancing Engine
 * ==========================================================
 *
 * Core service that orchestrates ML-powered exam generation for AeroBeacon.
 *
 * Responsibilities:
 *   1. Communicate with the Flask ML microservice for difficulty prediction
 *   2. Filter out recently-attempted questions per student
 *   3. Enforce the 30% Easy / 50% Medium / 20% Hard distribution
 *   4. Shuffle question order and MCQ options (Fisher-Yates)
 *   5. Fall back to rule-based difficulty assignment if ML is unavailable
 *
 * @module mlQuestionService
 */

const axios = require("axios");

// ── Configuration ───────────────────────────────────────────────────────────

/** Base URL of the Flask ML microservice */
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://127.0.0.1:5001";

/** Retry configuration for ML service calls */
const isTestEnv = process.env.NODE_ENV === "test";
const RETRY_CONFIG = {
  maxRetries: isTestEnv ? 1 : 3,
  baseDelayMs: isTestEnv ? 50 : 300,
  maxDelayMs: isTestEnv ? 200 : 3000,
  backoffMultiplier: 2,
};

/** Target difficulty distribution for a balanced exam */
const DIFFICULTY_RATIOS = {
  Easy: 0.30,
  Medium: 0.50,
  Hard: 0.20,
};

/** Number of recent exam sessions to check for question filtering */
const DEFAULT_RECENT_SESSIONS = 3;

/** Default exam length */
const DEFAULT_EXAM_LENGTH = 40;

// ── Simple Logger ───────────────────────────────────────────────────────────

const logger = {
  info: (msg, ...args) =>
    console.log(`[${new Date().toISOString()}] [INFO]  ${msg}`, ...args),
  warn: (msg, ...args) =>
    console.warn(`[${new Date().toISOString()}] [WARN]  ${msg}`, ...args),
  error: (msg, ...args) =>
    console.error(`[${new Date().toISOString()}] [ERROR] ${msg}`, ...args),
  debug: (msg, ...args) => {
    if (process.env.DEBUG === "true") {
      console.log(`[${new Date().toISOString()}] [DEBUG] ${msg}`, ...args);
    }
  },
};

// ── Utility Functions ───────────────────────────────────────────────────────

/**
 * Fisher-Yates (Knuth) shuffle — O(n), unbiased.
 *
 * Creates a new shuffled array; does NOT mutate the input.
 *
 * @param {Array} arr - Array to shuffle.
 * @returns {Array} A new array with elements in random order.
 * @throws {TypeError} If input is not an array.
 */
function fisherYatesShuffle(arr) {
  if (!Array.isArray(arr)) {
    throw new TypeError("fisherYatesShuffle expects an array");
  }
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/**
 * Shuffle MCQ options while preserving the correct answer mapping.
 *
 * @param {Object} question - Question object with `options` and `correctIndex`.
 * @returns {Object} New question object with shuffled options and updated correctIndex.
 * @throws {Error} If question is missing required fields.
 */
function shuffleOptions(question) {
  if (!question || !Array.isArray(question.options)) {
    throw new Error("Question must have an 'options' array");
  }
  if (
    typeof question.correctIndex !== "number" ||
    question.correctIndex < 0 ||
    question.correctIndex >= question.options.length
  ) {
    throw new Error(
      `Invalid correctIndex (${question.correctIndex}) for ${question.options.length} options`
    );
  }

  const correctOption = question.options[question.correctIndex];
  const shuffled = fisherYatesShuffle(question.options);
  const newCorrectIndex = shuffled.indexOf(correctOption);

  return {
    ...question,
    options: shuffled,
    correctIndex: newCorrectIndex,
  };
}

// ── ML Service Communication ────────────────────────────────────────────────

/**
 * Sleep for a specified number of milliseconds.
 *
 * @param {number} ms - Milliseconds to sleep.
 * @returns {Promise<void>}
 */
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Call the Flask ML service with exponential backoff retry logic.
 *
 * @param {string} endpoint - API endpoint (e.g., '/predict-batch').
 * @param {Object} payload - JSON payload to POST.
 * @param {Object} [retryOpts] - Override default retry config.
 * @returns {Promise<Object>} Response data from the ML service.
 * @throws {Error} After all retries are exhausted.
 */
async function callMlService(endpoint, payload, retryOpts = {}) {
  const config = { ...RETRY_CONFIG, ...retryOpts };
  let lastError;

  for (let attempt = 0; attempt <= config.maxRetries; attempt++) {
    try {
      const url = `${ML_SERVICE_URL}${endpoint}`;
      logger.debug(`ML request [attempt ${attempt + 1}]: POST ${url}`);

      const response = await axios.post(url, payload, {
        timeout: isTestEnv ? 1500 : 4000,
        headers: { "Content-Type": "application/json" },
      });

      return response.data;
    } catch (error) {
      lastError = error;
      const status = error.response?.status;
      const msg = error.message;

      // Don't retry on client errors (4xx) — our payload is bad
      if (status && status >= 400 && status < 500) {
        logger.error(`ML service returned ${status}: ${msg}`);
        throw error;
      }

      // Retry on network errors and 5xx
      if (attempt < config.maxRetries) {
        const delay = Math.min(
          config.baseDelayMs * Math.pow(config.backoffMultiplier, attempt),
          config.maxDelayMs
        );
        logger.warn(
          `ML service unavailable (attempt ${attempt + 1}/${config.maxRetries + 1}): ${msg}. Retrying in ${delay}ms...`
        );
        await sleep(delay);
      }
    }
  }

  logger.error(
    `ML service unreachable after ${config.maxRetries + 1} attempts: ${lastError.message}`
  );
  throw lastError;
}

/**
 * Predict difficulty for a batch of questions via the ML service.
 *
 * @param {Object[]} questions - Array of question objects from the database.
 * @returns {Promise<Object[]>} Array of { difficulty, confidence, probabilities }.
 */
async function predictDifficultyBatch(questions) {
  if (!questions.length) return [];

  const CHUNK_SIZE = 200;
  const allResults = [];

  for (let i = 0; i < questions.length; i += CHUNK_SIZE) {
    const chunk = questions.slice(i, i + CHUNK_SIZE);
    const payload = {
      questions: chunk.map((q) => ({
        text_length: (q.text || "").length,
        num_options: (q.options || []).length || 4,
        avg_time_taken: q.avgTimeTaken || 60,
        past_accuracy: q.pastAccuracy ?? 0.5,
      })),
    };
    const chunkResults = await callMlService("/predict-batch", payload);
    allResults.push(...chunkResults);
  }

  return allResults;
}

// ── Rule-Based Fallback ─────────────────────────────────────────────────────

/**
 * Assign difficulty using rule-based heuristics (no ML needed).
 *
 * Used when the Flask ML service is unreachable. Implements a simple
 * decision tree based on past_accuracy and avg_time_taken.
 *
 * Rules:
 *   - past_accuracy >= 0.70 AND avg_time < 45s  → Easy
 *   - past_accuracy <= 0.40 OR avg_time >= 75s   → Hard
 *   - Everything else                             → Medium
 *
 * @param {Object} question - Question object.
 * @returns {Object} { difficulty, confidence, source }
 */
function ruleBasedDifficultyPredict(question) {
  const accuracy = question.pastAccuracy ?? 0.5;
  const time = question.avgTimeTaken || 60;
  const textLen = (question.text || "").length;

  let difficulty = "Medium";
  let confidence = 0.5;

  if (accuracy >= 0.7 && time < 45) {
    difficulty = "Easy";
    confidence = 0.7 + (accuracy - 0.7) * 2; // Higher accuracy → higher confidence
  } else if (accuracy <= 0.4 || time >= 75) {
    difficulty = "Hard";
    confidence = 0.6 + (0.4 - Math.min(accuracy, 0.4)) * 2;
  } else {
    difficulty = "Medium";
    confidence = 0.6;
  }

  // Text length bonus (longer questions tend to be harder)
  if (textLen > 250 && difficulty !== "Hard") {
    confidence *= 0.9; // reduce confidence if text is long but not classified hard
  }

  return {
    difficulty,
    confidence: Math.min(Math.round(confidence * 1000) / 1000, 1.0),
    source: "rule_based_fallback",
  };
}

// ── Question Filtering ──────────────────────────────────────────────────────

/**
 * Remove questions the student has recently attempted.
 *
 * Examines the student's last N exam sessions and excludes any
 * questions that appeared in them.
 *
 * @param {Object[]} pool - Full question pool.
 * @param {Object[]} studentHistory - Array of past ExamSession documents.
 * @param {number} [lastN=3] - Number of recent sessions to look back.
 * @returns {Object[]} Filtered question pool.
 */
function filterRecentQuestions(pool, studentHistory, lastN = DEFAULT_RECENT_SESSIONS) {
  if (!Array.isArray(studentHistory) || studentHistory.length === 0) {
    logger.debug("No student history — returning full pool");
    return pool;
  }

  const recentIds = new Set();
  const recentSessions = studentHistory.slice(-lastN);

  for (const session of recentSessions) {
    if (Array.isArray(session.questionIds)) {
      for (const id of session.questionIds) {
        recentIds.add(id.toString());
      }
    }
  }

  const filtered = pool.filter((q) => {
    const qId = (q._id || q.id || "").toString();
    return !recentIds.has(qId);
  });

  logger.info(
    `Filtered recently-attempted questions: ${pool.length} → ${filtered.length} (excluded ${recentIds.size} IDs from ${recentSessions.length} sessions)`
  );

  // If filtering removes too many, warn and return what we have
  if (filtered.length === 0 && pool.length > 0) {
    logger.warn(
      "All questions were recently attempted! Returning full pool to avoid empty exam."
    );
    return pool;
  }

  return filtered;
}

// ── Stratified Sampling ─────────────────────────────────────────────────────

/**
 * Sample questions to match the target 30/50/20 difficulty distribution.
 *
 * If a category has fewer questions than needed, the shortfall is filled
 * from the remaining pool to ensure the exam is the right length.
 *
 * @param {Object[]} questions - Questions with `predictedDifficulty` set.
 * @param {number} totalCount - Target number of questions.
 * @returns {Object[]} Stratified sample of questions.
 */
function stratifiedSample(questions, totalCount) {
  const nEasy = Math.round(totalCount * DIFFICULTY_RATIOS.Easy);
  const nHard = Math.round(totalCount * DIFFICULTY_RATIOS.Hard);
  const nMedium = totalCount - nEasy - nHard;

  logger.info(
    `Target distribution: Easy=${nEasy}, Medium=${nMedium}, Hard=${nHard} (total=${totalCount})`
  );

  // Bucket questions by predicted difficulty
  const buckets = { Easy: [], Medium: [], Hard: [] };
  for (const q of questions) {
    const diff = q.predictedDifficulty || "Medium";
    if (buckets[diff]) {
      buckets[diff].push(q);
    } else {
      buckets.Medium.push(q); // fallback bucket
    }
  }

  // Shuffle each bucket
  const shuffledBuckets = {
    Easy: fisherYatesShuffle(buckets.Easy),
    Medium: fisherYatesShuffle(buckets.Medium),
    Hard: fisherYatesShuffle(buckets.Hard),
  };

  // Pick from each bucket
  const pick = (arr, n) => arr.slice(0, n);
  let selected = [
    ...pick(shuffledBuckets.Easy, nEasy),
    ...pick(shuffledBuckets.Medium, nMedium),
    ...pick(shuffledBuckets.Hard, nHard),
  ];

  // Log actual distribution
  const actualDist = {
    Easy: Math.min(shuffledBuckets.Easy.length, nEasy),
    Medium: Math.min(shuffledBuckets.Medium.length, nMedium),
    Hard: Math.min(shuffledBuckets.Hard.length, nHard),
  };
  logger.info(
    `Actual distribution: Easy=${actualDist.Easy}, Medium=${actualDist.Medium}, Hard=${actualDist.Hard}`
  );

  // Fill shortfall from remaining pool
  if (selected.length < totalCount) {
    const usedIds = new Set(
      selected.map((q) => (q._id || q.id || "").toString())
    );
    const remaining = questions.filter(
      (q) => !usedIds.has((q._id || q.id || "").toString())
    );
    const shuffledRemaining = fisherYatesShuffle(remaining);
    const shortfall = totalCount - selected.length;

    logger.warn(
      `Shortfall of ${shortfall} questions. Filling from remaining pool (${remaining.length} available).`
    );

    selected = selected.concat(shuffledRemaining.slice(0, shortfall));
  }

  return selected.slice(0, totalCount);
}

// ── Main Exam Builder ───────────────────────────────────────────────────────

/**
 * Build a balanced mock exam for a student.
 *
 * Full pipeline:
 *   1. Filter out recently-attempted questions
 *   2. Predict difficulty via ML (with rule-based fallback)
 *   3. Stratified sampling to enforce 30/50/20 distribution
 *   4. Shuffle question order
 *   5. Shuffle MCQ options within each question
 *
 * @param {string} studentId - MongoDB ObjectId of the student.
 * @param {Object[]} pool - Full question pool from the database.
 * @param {Object[]} studentHistory - Array of past ExamSession documents.
 * @param {number} [totalQuestions=40] - Number of questions in the exam.
 * @returns {Promise<Object>} { questions, metadata }
 */
async function buildBalancedExam(
  studentId,
  pool,
  studentHistory = [],
  totalQuestions = DEFAULT_EXAM_LENGTH
) {
  if (!Array.isArray(pool) || pool.length === 0) {
    throw new Error("Question pool is empty or invalid");
  }

  if (totalQuestions < 1) {
    throw new Error("totalQuestions must be at least 1");
  }

  logger.info(
    `Building exam for student ${studentId}: ${totalQuestions} questions from pool of ${pool.length}`
  );

  // 1. Filter recently-attempted questions
  const filtered = filterRecentQuestions(pool, studentHistory);

  if (filtered.length < totalQuestions) {
    logger.warn(
      `Pool (${filtered.length}) is smaller than requested exam length (${totalQuestions}). Exam will be shorter.`
    );
  }

  // 2. Predict difficulty
  let predictions;
  let mlSource = "ml_service";

  try {
    predictions = await predictDifficultyBatch(filtered);
    logger.info("ML predictions received successfully");
  } catch (error) {
    logger.warn(
      `ML service unavailable: ${error.message}. Using rule-based fallback.`
    );
    predictions = filtered.map((q) => ruleBasedDifficultyPredict(q));
    mlSource = "rule_based_fallback";
  }

  // 3. Attach predictions to questions
  const questionsWithDifficulty = filtered.map((q, i) => {
    // Create a plain object copy to avoid Mongoose document issues
    const plain =
      typeof q.toObject === "function" ? q.toObject() : { ...q };
    plain.predictedDifficulty = predictions[i]?.difficulty || "Medium";
    plain.predictionConfidence = predictions[i]?.confidence || 0;
    return plain;
  });

  // 4. Stratified sampling
  const actualCount = Math.min(totalQuestions, questionsWithDifficulty.length);
  const sampled = stratifiedSample(questionsWithDifficulty, actualCount);

  // 5. Shuffle order and options
  const shuffled = fisherYatesShuffle(sampled);
  const final = shuffled.map((q) => {
    try {
      return shuffleOptions(q);
    } catch (e) {
      logger.warn(`Could not shuffle options for question ${q._id}: ${e.message}`);
      return q;
    }
  });

  // 6. Compute metadata
  const breakdown = { easy: 0, medium: 0, hard: 0 };
  for (const q of final) {
    const d = (q.predictedDifficulty || "Medium").toLowerCase();
    if (breakdown.hasOwnProperty(d)) {
      breakdown[d]++;
    }
  }

  const metadata = {
    studentId,
    totalQuestions: final.length,
    requestedQuestions: totalQuestions,
    poolSize: pool.length,
    filteredPoolSize: filtered.length,
    difficultyBreakdown: breakdown,
    predictionSource: mlSource,
    generatedAt: new Date().toISOString(),
  };

  logger.info(
    `Exam built: ${final.length} questions [E:${breakdown.easy} M:${breakdown.medium} H:${breakdown.hard}] via ${mlSource}`
  );

  return { questions: final, metadata };
}

// ── Health Check ────────────────────────────────────────────────────────────

/**
 * Check if the ML Flask service is reachable.
 *
 * @returns {Promise<Object>} Health status from the ML service.
 */
async function checkMlServiceHealth() {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/health`, {
      timeout: 5000,
    });
    return { available: true, ...response.data };
  } catch (error) {
    return {
      available: false,
      error: error.message,
      fallback: "rule_based",
    };
  }
}

// ── Exports ─────────────────────────────────────────────────────────────────

module.exports = {
  buildBalancedExam,
  fisherYatesShuffle,
  shuffleOptions,
  stratifiedSample,
  filterRecentQuestions,
  predictDifficultyBatch,
  ruleBasedDifficultyPredict,
  checkMlServiceHealth,
  // Exposed for testing
  _internals: {
    callMlService,
    sleep,
    DIFFICULTY_RATIOS,
    RETRY_CONFIG,
    ML_SERVICE_URL,
  },
};
