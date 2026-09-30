const mongoose = require("mongoose");

/**
 * Attempt Schema — A single student response to a question.
 *
 * Records whether the student answered correctly, which option they chose,
 * and how long they took. Used for:
 *   - Computing past_accuracy for ML retraining
 *   - Filtering recently-attempted questions from future exams
 *   - Student performance analytics
 *
 * @property {ObjectId} studentId        - Reference to the student
 * @property {ObjectId} questionId       - Reference to the question
 * @property {ObjectId} examId           - Reference to the exam session
 * @property {number}   selectedOption   - Zero-based index of chosen option
 * @property {boolean}  isCorrect        - Whether the answer was correct
 * @property {number}   timeTakenSeconds - Time spent on this question
 * @property {Date}     attemptDate      - When the attempt was made
 */
const attemptSchema = new mongoose.Schema(
  {
    studentId: {
      type: mongoose.Schema.Types.ObjectId,
      required: [true, "Student ID is required"],
      index: true,
    },
    questionId: {
      type: mongoose.Schema.Types.ObjectId,
      required: [true, "Question ID is required"],
      ref: "Question",
      index: true,
    },
    examId: {
      type: mongoose.Schema.Types.ObjectId,
      required: [true, "Exam ID is required"],
      ref: "ExamSession",
      index: true,
    },
    selectedOption: {
      type: Number,
      required: [true, "Selected option is required"],
      min: [0, "Selected option must be >= 0"],
    },
    isCorrect: {
      type: Boolean,
      required: [true, "isCorrect flag is required"],
    },
    timeTakenSeconds: {
      type: Number,
      required: [true, "Time taken is required"],
      min: [0, "Time taken must be >= 0"],
      max: [3600, "Time taken cannot exceed 3600 seconds"],
    },
    attemptDate: {
      type: Date,
      default: Date.now,
    },
  },
  {
    timestamps: true,
  }
);

// Compound index for efficient lookups during retraining
attemptSchema.index({ studentId: 1, attemptDate: -1 });
attemptSchema.index({ questionId: 1, isCorrect: 1 });

module.exports = mongoose.model("Attempt", attemptSchema);
