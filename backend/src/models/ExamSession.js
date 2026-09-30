const mongoose = require("mongoose");

/**
 * ExamSession Schema — A complete mock exam taken by a student.
 *
 * Tracks which questions were served, the difficulty breakdown, timing,
 * and final score. Referenced by Attempt documents.
 *
 * @property {ObjectId}   studentId           - Reference to the student
 * @property {ObjectId[]} questionIds         - Ordered list of served question IDs
 * @property {Date}       startedAt           - When the exam began
 * @property {Date}       submittedAt         - When the exam was submitted
 * @property {number}     score               - Number of correct answers
 * @property {number}     totalQuestions      - Total questions in the exam
 * @property {Object}     difficultyBreakdown - Count per difficulty level
 * @property {string}     status              - Current exam state
 */
const examSessionSchema = new mongoose.Schema(
  {
    studentId: {
      type: mongoose.Schema.Types.ObjectId,
      required: [true, "Student ID is required"],
      index: true,
    },
    questionIds: {
      type: [mongoose.Schema.Types.ObjectId],
      required: [true, "Question IDs are required"],
      ref: "Question",
      validate: {
        validator: (arr) => arr.length > 0,
        message: "Exam must contain at least one question",
      },
    },
    startedAt: {
      type: Date,
      default: Date.now,
    },
    submittedAt: {
      type: Date,
      default: null,
    },
    score: {
      type: Number,
      default: 0,
      min: [0, "Score must be >= 0"],
    },
    totalQuestions: {
      type: Number,
      required: [true, "Total questions count is required"],
      min: [1, "Must have at least 1 question"],
    },
    difficultyBreakdown: {
      easy: { type: Number, default: 0, min: 0 },
      medium: { type: Number, default: 0, min: 0 },
      hard: { type: Number, default: 0, min: 0 },
    },
    status: {
      type: String,
      enum: ["in_progress", "submitted", "expired"],
      default: "in_progress",
    },
  },
  {
    timestamps: true,
    toJSON: { virtuals: true },
    toObject: { virtuals: true },
  }
);

/**
 * Virtual: exam duration in seconds (null if not yet submitted).
 */
examSessionSchema.virtual("durationSeconds").get(function () {
  if (!this.submittedAt || !this.startedAt) return null;
  return Math.round((this.submittedAt - this.startedAt) / 1000);
});

/**
 * Virtual: score percentage (null if exam not submitted).
 */
examSessionSchema.virtual("scorePercent").get(function () {
  if (this.totalQuestions === 0) return 0;
  return Math.round((this.score / this.totalQuestions) * 100);
});

module.exports = mongoose.model("ExamSession", examSessionSchema);
