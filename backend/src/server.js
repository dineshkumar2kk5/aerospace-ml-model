/**
 * server.js — AeroBeacon Express API Server
 * ==========================================
 *
 * Full Express server exposing the ML-based exam generation API,
 * question management, ML prediction proxy, and offline question pool.
 *
 * Endpoints:
 *   POST /api/exam/generate        - Generate a balanced mock exam (30/50/20)
 *   GET  /api/health               - Backend health check
 *   GET  /api/ml-status            - ML service connectivity status
 *   GET  /api/questions            - Query & browse question bank
 *   POST /api/questions/predict    - Predict difficulty for a question
 *   GET  /api/stats                - Aggregate statistics on question pool & ML
 */

require("dotenv").config();
const fs = require("fs");
const path = require("path");
const express = require("express");
const mongoose = require("mongoose");
const {
  buildBalancedExam,
  checkMlServiceHealth,
  predictDifficultyBatch,
  ruleBasedDifficultyPredict,
} = require("./services/mlQuestionService");
const Question = require("./models/Question");
const ExamSession = require("./models/ExamSession");

const app = express();
app.use(express.json());

// Enable CORS for local development & frontend flexibility
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header(
    "Access-Control-Allow-Headers",
    "Origin, X-Requested-With, Content-Type, Accept"
  );
  res.header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
  if (req.method === "OPTIONS") return res.sendStatus(200);
  next();
});

// Serve frontend static assets from public/
const publicDir = path.join(__dirname, "../public");
if (fs.existsSync(publicDir)) {
  app.use(express.static(publicDir));
}

const PORT = process.env.PORT || 3001;
const MONGO_URI =
  process.env.MONGO_URI || "mongodb://localhost:27017/aerobeacon";

// ── Offline Question Pool Helper ─────────────────────────────────────────────

function getOfflineQuestions(topic) {
  const filePath = path.join(__dirname, "../data/questions.json");
  if (!fs.existsSync(filePath)) return [];
  try {
    const raw = fs.readFileSync(filePath, "utf-8");
    const list = JSON.parse(raw);
    return topic ? list.filter((q) => q.topic === topic) : list;
  } catch (err) {
    console.error(`[ERROR] Failed reading offline questions: ${err.message}`);
    return [];
  }
}

// ── Database Connection ─────────────────────────────────────────────────────

async function connectDB() {
  try {
    await mongoose.connect(MONGO_URI, { serverSelectionTimeoutMS: 2000 });
    console.log(`[INFO] Connected to MongoDB: ${MONGO_URI}`);
  } catch (err) {
    console.warn(`[WARN] MongoDB connection unavailable (${err.message}). Using high-performance offline JSON pool.`);
  }
}

// ── Routes ──────────────────────────────────────────────────────────────────

/**
 * GET /api/health — Backend health check.
 */
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    service: "aerobeacon-backend",
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
    mongoConnected: mongoose.connection.readyState === 1,
    offlinePoolAvailable: fs.existsSync(path.join(__dirname, "../data/questions.json")),
  });
});

/**
 * GET /api/ml-status — Check if the ML service is reachable.
 */
app.get("/api/ml-status", async (req, res) => {
  try {
    const status = await checkMlServiceHealth();
    res.json(status);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * GET /api/stats — Summary metrics of the question pool & model.
 */
app.get("/api/stats", async (req, res) => {
  try {
    let pool;
    if (mongoose.connection.readyState === 1) {
      pool = await Question.find({}).lean();
    } else {
      pool = getOfflineQuestions();
    }

    const byDifficulty = { Easy: 0, Medium: 0, Hard: 0 };
    const byTopic = {};
    let icJoshiCount = 0;

    for (const q of pool) {
      const d = q.difficulty || "Medium";
      byDifficulty[d] = (byDifficulty[d] || 0) + 1;

      const t = q.topic || "General";
      byTopic[t] = (byTopic[t] || 0) + 1;

      if ((q.id && q.id.startsWith("joshi")) || q.topic) {
        icJoshiCount++;
      }
    }

    const mlStatus = await checkMlServiceHealth().catch(() => ({ available: false }));

    res.json({
      totalQuestions: pool.length,
      difficultyBreakdown: byDifficulty,
      topicBreakdown: byTopic,
      icJoshiMeteorologyQuestions: icJoshiCount,
      mlService: mlStatus,
      targetDistribution: { Easy: "30%", Medium: "50%", Hard: "20%" },
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * GET /api/questions — Browse the question bank with filtering.
 */
app.get("/api/questions", async (req, res) => {
  try {
    const { topic, difficulty, search, limit = 50, offset = 0 } = req.query;

    let list;
    if (mongoose.connection.readyState === 1) {
      const query = {};
      if (topic) query.topic = topic;
      if (difficulty) query.difficulty = difficulty;
      if (search) query.text = { $regex: search, $options: "i" };
      list = await Question.find(query)
        .skip(Number(offset))
        .limit(Number(limit))
        .lean();
    } else {
      list = getOfflineQuestions(topic);
      if (difficulty) list = list.filter((q) => q.difficulty === difficulty);
      if (search) {
        const s = search.toLowerCase();
        list = list.filter((q) => q.text.toLowerCase().includes(s));
      }
      list = list.slice(Number(offset), Number(offset) + Number(limit));
    }

    res.json({
      count: list.length,
      limit: Number(limit),
      offset: Number(offset),
      questions: list,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * POST /api/questions/predict — Predict difficulty on-the-fly.
 */
app.post("/api/questions/predict", async (req, res) => {
  try {
    const { text = "", options = [], avgTimeTaken = 60, pastAccuracy = 0.5 } = req.body;
    const text_length = text.length || 50;
    const num_options = Array.isArray(options) && options.length > 0 ? options.length : 4;

    const mlResult = await predictDifficultyBatch([
      { text_length, num_options, avg_time_taken: avgTimeTaken, past_accuracy: pastAccuracy },
    ]);

    if (mlResult && mlResult.length > 0) {
      res.json(mlResult[0]);
    } else {
      const fallback = ruleBasedDifficultyPredict({
        text_length,
        num_options,
        avg_time_taken: avgTimeTaken,
        past_accuracy: pastAccuracy,
      });
      res.json(fallback);
    }
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * POST /api/exam/generate — Generate a balanced mock exam.
 *
 * Body:
 *   {
 *     "studentId": "pilot_001",
 *     "totalQuestions": 40,
 *     "topic": "Meteorology"  // optional
 *   }
 */
app.post("/api/exam/generate", async (req, res) => {
  try {
    const { studentId, totalQuestions = 40, topic } = req.body;

    if (!studentId) {
      return res.status(400).json({ error: "studentId is required" });
    }

    // Fetch question pool
    let pool;
    if (mongoose.connection.readyState === 1) {
      const query = topic ? { topic } : {};
      pool = await Question.find(query).lean();
    } else {
      pool = getOfflineQuestions(topic);
    }

    if (!pool || pool.length === 0) {
      return res.status(404).json({
        error: "No questions available in the pool",
        hint: "Ensure backend/data/questions.json exists or database is seeded",
      });
    }

    // Fetch student history
    let history = [];
    if (mongoose.connection.readyState === 1) {
      history = await ExamSession.find({ studentId })
        .sort({ startedAt: -1 })
        .limit(5)
        .lean();
    }

    // Build balanced exam
    const result = await buildBalancedExam(
      studentId,
      pool,
      history,
      Number(totalQuestions)
    );

    res.json(result);
  } catch (err) {
    console.error(`[ERROR] Exam generation failed: ${err.message}`);
    res.status(500).json({ error: "Failed to generate exam", details: err.message });
  }
});

// ── Error Handling ──────────────────────────────────────────────────────────

app.use((err, req, res, _next) => {
  console.error(`[ERROR] Unhandled: ${err.message}`);
  res.status(500).json({ error: "Internal server error" });
});

// ── Start ───────────────────────────────────────────────────────────────────

async function start() {
  await connectDB();
  app.listen(PORT, () => {
    console.log(`[INFO] 🚀 AeroBeacon backend running on http://localhost:${PORT}`);
  });
}

// Only start if run directly (not during tests)
if (require.main === module) {
  start();
}

module.exports = { app };
