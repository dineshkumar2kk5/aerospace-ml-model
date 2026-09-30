const mongoose = require("mongoose");

/**
 * Question Schema — A single DGCA exam question.
 *
 * @property {string}   text          - Full question text
 * @property {string}   topic         - DGCA syllabus topic category
 * @property {string[]} options       - Array of MCQ answer choices
 * @property {number}   correctIndex  - Zero-based index of the correct option
 * @property {string}   difficulty    - Ground-truth difficulty (Easy/Medium/Hard)
 * @property {number}   avgTimeTaken  - Average time (seconds) students spend on this
 * @property {number}   pastAccuracy  - Historical correct-answer rate (0.0–1.0)
 * @property {Date}     createdAt     - When the question was added
 * @property {Date}     updatedAt     - Last modification timestamp
 */
const questionSchema = new mongoose.Schema(
  {
    text: {
      type: String,
      required: [true, "Question text is required"],
      trim: true,
      minlength: [10, "Question text must be at least 10 characters"],
    },
    topic: {
      type: String,
      required: [true, "Topic is required"],
      trim: true,
      enum: {
        values: [
          "Meteorology",
          "Navigation",
          "Air Regulations",
          "Technical General",
          "Radio Aids",
          "Human Performance",
          "Flight Planning",
          "Aircraft Systems",
        ],
        message: "{VALUE} is not a valid topic",
      },
    },
    options: {
      type: [String],
      required: [true, "Options are required"],
      validate: {
        validator: (arr) => arr.length >= 2 && arr.length <= 6,
        message: "Must have between 2 and 6 options",
      },
    },
    correctIndex: {
      type: Number,
      required: [true, "Correct index is required"],
      min: [0, "Correct index must be >= 0"],
    },
    difficulty: {
      type: String,
      enum: {
        values: ["Easy", "Medium", "Hard"],
        message: "{VALUE} is not a valid difficulty level",
      },
      default: "Medium",
    },
    avgTimeTaken: {
      type: Number,
      default: 60,
      min: [1, "Average time must be at least 1 second"],
      max: [600, "Average time cannot exceed 600 seconds"],
    },
    pastAccuracy: {
      type: Number,
      default: 0.5,
      min: [0, "Past accuracy must be >= 0"],
      max: [1, "Past accuracy must be <= 1"],
    },
  },
  {
    timestamps: true,
    toJSON: { virtuals: true },
    toObject: { virtuals: true },
  }
);

// Indexes for common query patterns
questionSchema.index({ topic: 1, difficulty: 1 });
questionSchema.index({ pastAccuracy: 1 });

/**
 * Virtual: predicted difficulty (set at runtime by ML service, not persisted).
 */
questionSchema.virtual("predictedDifficulty").get(function () {
  return this._predictedDifficulty;
});
questionSchema.virtual("predictedDifficulty").set(function (val) {
  this._predictedDifficulty = val;
});

module.exports = mongoose.model("Question", questionSchema);
