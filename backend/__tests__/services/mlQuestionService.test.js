/**
 * mlQuestionService.test.js — Unit Tests
 * ========================================
 *
 * Tests for the ML-based question balancing engine.
 * Covers: Fisher-Yates shuffle, option shuffling, question filtering,
 *         stratified sampling, rule-based fallback, and exam building.
 */

const {
  fisherYatesShuffle,
  shuffleOptions,
  stratifiedSample,
  filterRecentQuestions,
  ruleBasedDifficultyPredict,
  buildBalancedExam,
} = require("../../src/services/mlQuestionService");

// ── Helper: Generate mock questions ─────────────────────────────────────────

/**
 * Create a mock question object for testing.
 *
 * @param {string} id - Question ID.
 * @param {string} [difficulty="Medium"] - Predicted difficulty.
 * @param {Object} [overrides={}] - Additional field overrides.
 * @returns {Object} Mock question.
 */
function createMockQuestion(id, difficulty = "Medium", overrides = {}) {
  return {
    _id: id,
    text: `Question ${id} text for testing purposes that varies in length`,
    topic: "Navigation",
    options: ["Option A", "Option B", "Option C", "Option D"],
    correctIndex: 0,
    difficulty,
    predictedDifficulty: difficulty,
    avgTimeTaken: difficulty === "Easy" ? 30 : difficulty === "Hard" ? 100 : 60,
    pastAccuracy: difficulty === "Easy" ? 0.85 : difficulty === "Hard" ? 0.25 : 0.55,
    ...overrides,
  };
}

/**
 * Generate a pool of N questions with a given difficulty distribution.
 */
function createMockPool(counts = { Easy: 30, Medium: 50, Hard: 20 }) {
  const pool = [];
  let id = 1;
  for (const [difficulty, count] of Object.entries(counts)) {
    for (let i = 0; i < count; i++) {
      pool.push(createMockQuestion(`q${id++}`, difficulty));
    }
  }
  return pool;
}

// ═══════════════════════════════════════════════════════════════════════════
// Fisher-Yates Shuffle
// ═══════════════════════════════════════════════════════════════════════════

describe("fisherYatesShuffle", () => {
  test("returns a new array (no mutation)", () => {
    const original = [1, 2, 3, 4, 5];
    const originalCopy = [...original];
    fisherYatesShuffle(original);
    expect(original).toEqual(originalCopy);
  });

  test("preserves all elements (same length, same values)", () => {
    const arr = [10, 20, 30, 40, 50];
    const shuffled = fisherYatesShuffle(arr);
    expect(shuffled).toHaveLength(arr.length);
    expect(shuffled.sort()).toEqual(arr.sort());
  });

  test("handles empty array", () => {
    expect(fisherYatesShuffle([])).toEqual([]);
  });

  test("handles single-element array", () => {
    expect(fisherYatesShuffle([42])).toEqual([42]);
  });

  test("throws TypeError for non-array input", () => {
    expect(() => fisherYatesShuffle("not an array")).toThrow(TypeError);
    expect(() => fisherYatesShuffle(null)).toThrow(TypeError);
    expect(() => fisherYatesShuffle(123)).toThrow(TypeError);
  });

  test("produces different orders across many runs (statistical)", () => {
    const arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const results = new Set();
    for (let i = 0; i < 50; i++) {
      results.add(JSON.stringify(fisherYatesShuffle(arr)));
    }
    // With 10 elements, 50 shuffles should produce many distinct orderings
    expect(results.size).toBeGreaterThan(10);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Shuffle Options
// ═══════════════════════════════════════════════════════════════════════════

describe("shuffleOptions", () => {
  test("preserves the correct answer after shuffling", () => {
    const q = createMockQuestion("q1");
    const correctAnswer = q.options[q.correctIndex];

    // Run 20 times to be statistically confident
    for (let i = 0; i < 20; i++) {
      const shuffled = shuffleOptions(q);
      expect(shuffled.options[shuffled.correctIndex]).toBe(correctAnswer);
    }
  });

  test("preserves all options", () => {
    const q = createMockQuestion("q1");
    const shuffled = shuffleOptions(q);
    expect(shuffled.options.sort()).toEqual(q.options.sort());
  });

  test("does not mutate the original question", () => {
    const q = createMockQuestion("q1");
    const originalOptions = [...q.options];
    const originalIndex = q.correctIndex;

    shuffleOptions(q);

    expect(q.options).toEqual(originalOptions);
    expect(q.correctIndex).toBe(originalIndex);
  });

  test("throws on missing options", () => {
    expect(() => shuffleOptions({ correctIndex: 0 })).toThrow();
    expect(() => shuffleOptions(null)).toThrow();
  });

  test("throws on invalid correctIndex", () => {
    expect(() =>
      shuffleOptions({
        options: ["A", "B", "C"],
        correctIndex: 5,
      })
    ).toThrow("Invalid correctIndex");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Filter Recent Questions
// ═══════════════════════════════════════════════════════════════════════════

describe("filterRecentQuestions", () => {
  test("removes recently-attempted questions", () => {
    const pool = createMockPool({ Easy: 5, Medium: 5, Hard: 5 });
    const history = [
      { questionIds: ["q1", "q2", "q3"] },
      { questionIds: ["q4", "q5"] },
    ];

    const filtered = filterRecentQuestions(pool, history, 3);
    const filteredIds = filtered.map((q) => q._id);

    expect(filteredIds).not.toContain("q1");
    expect(filteredIds).not.toContain("q2");
    expect(filteredIds).not.toContain("q3");
    expect(filteredIds).not.toContain("q4");
    expect(filteredIds).not.toContain("q5");
    expect(filtered).toHaveLength(10);
  });

  test("returns full pool when history is empty", () => {
    const pool = createMockPool({ Easy: 5, Medium: 5, Hard: 5 });
    const filtered = filterRecentQuestions(pool, [], 3);
    expect(filtered).toHaveLength(15);
  });

  test("returns full pool when history is undefined", () => {
    const pool = createMockPool({ Easy: 5, Medium: 5, Hard: 5 });
    const filtered = filterRecentQuestions(pool, undefined, 3);
    expect(filtered).toHaveLength(15);
  });

  test("only looks back lastN sessions", () => {
    const pool = createMockPool({ Easy: 10, Medium: 0, Hard: 0 });
    const history = [
      { questionIds: ["q1"] },      // session 1 (oldest — should NOT be filtered)
      { questionIds: ["q2", "q3"] }, // session 2
      { questionIds: ["q4"] },       // session 3 (newest)
    ];

    // Only look back 2 sessions → q1 should remain
    const filtered = filterRecentQuestions(pool, history, 2);
    const filteredIds = filtered.map((q) => q._id);

    expect(filteredIds).toContain("q1");
    expect(filteredIds).not.toContain("q2");
    expect(filteredIds).not.toContain("q3");
    expect(filteredIds).not.toContain("q4");
  });

  test("returns full pool if all questions were recently attempted", () => {
    const pool = [
      createMockQuestion("q1"),
      createMockQuestion("q2"),
    ];
    const history = [{ questionIds: ["q1", "q2"] }];

    // Should return full pool with a warning, not empty
    const filtered = filterRecentQuestions(pool, history, 3);
    expect(filtered).toHaveLength(2);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Stratified Sampling
// ═══════════════════════════════════════════════════════════════════════════

describe("stratifiedSample", () => {
  test("returns correct total count", () => {
    const pool = createMockPool({ Easy: 30, Medium: 50, Hard: 20 });
    const sampled = stratifiedSample(pool, 40);
    expect(sampled).toHaveLength(40);
  });

  test("enforces ~30/50/20 distribution", () => {
    const pool = createMockPool({ Easy: 50, Medium: 80, Hard: 40 });
    const sampled = stratifiedSample(pool, 100);

    const distribution = { Easy: 0, Medium: 0, Hard: 0 };
    sampled.forEach((q) => {
      distribution[q.predictedDifficulty]++;
    });

    expect(distribution.Easy).toBe(30);   // 30%
    expect(distribution.Medium).toBe(50); // 50%
    expect(distribution.Hard).toBe(20);   // 20%
  });

  test("handles pool smaller than requested count", () => {
    const pool = createMockPool({ Easy: 3, Medium: 5, Hard: 2 });
    const sampled = stratifiedSample(pool, 10);
    expect(sampled).toHaveLength(10);
  });

  test("fills shortfall from remaining pool when a category is empty", () => {
    // Only easy questions, no medium or hard
    const pool = createMockPool({ Easy: 40, Medium: 0, Hard: 0 });
    const sampled = stratifiedSample(pool, 20);
    expect(sampled).toHaveLength(20);
  });

  test("handles request for 1 question", () => {
    const pool = createMockPool({ Easy: 5, Medium: 5, Hard: 5 });
    const sampled = stratifiedSample(pool, 1);
    expect(sampled).toHaveLength(1);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Rule-Based Fallback
// ═══════════════════════════════════════════════════════════════════════════

describe("ruleBasedDifficultyPredict", () => {
  test("classifies high-accuracy, fast questions as Easy", () => {
    const q = createMockQuestion("q1", "Easy", {
      pastAccuracy: 0.9,
      avgTimeTaken: 25,
    });
    const result = ruleBasedDifficultyPredict(q);
    expect(result.difficulty).toBe("Easy");
    expect(result.source).toBe("rule_based_fallback");
  });

  test("classifies low-accuracy questions as Hard", () => {
    const q = createMockQuestion("q1", "Hard", {
      pastAccuracy: 0.2,
      avgTimeTaken: 100,
    });
    const result = ruleBasedDifficultyPredict(q);
    expect(result.difficulty).toBe("Hard");
  });

  test("classifies slow questions as Hard", () => {
    const q = createMockQuestion("q1", "Medium", {
      pastAccuracy: 0.55,
      avgTimeTaken: 90,
    });
    const result = ruleBasedDifficultyPredict(q);
    expect(result.difficulty).toBe("Hard");
  });

  test("classifies mid-range questions as Medium", () => {
    const q = createMockQuestion("q1", "Medium", {
      pastAccuracy: 0.55,
      avgTimeTaken: 55,
    });
    const result = ruleBasedDifficultyPredict(q);
    expect(result.difficulty).toBe("Medium");
  });

  test("returns confidence between 0 and 1", () => {
    const questions = [
      createMockQuestion("q1", "Easy", { pastAccuracy: 0.9, avgTimeTaken: 20 }),
      createMockQuestion("q2", "Medium", { pastAccuracy: 0.55, avgTimeTaken: 55 }),
      createMockQuestion("q3", "Hard", { pastAccuracy: 0.15, avgTimeTaken: 110 }),
    ];

    for (const q of questions) {
      const result = ruleBasedDifficultyPredict(q);
      expect(result.confidence).toBeGreaterThanOrEqual(0);
      expect(result.confidence).toBeLessThanOrEqual(1);
    }
  });

  test("handles missing fields gracefully", () => {
    const result = ruleBasedDifficultyPredict({});
    expect(["Easy", "Medium", "Hard"]).toContain(result.difficulty);
    expect(result.source).toBe("rule_based_fallback");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Build Balanced Exam (Integration-style, mocking ML service)
// ═══════════════════════════════════════════════════════════════════════════

describe("buildBalancedExam", () => {
  test("builds exam with valid metadata when generated", async () => {
    const pool = createMockPool({ Easy: 30, Medium: 50, Hard: 20 });
    const result = await buildBalancedExam("student123", pool, [], 20);

    expect(result).toHaveProperty("questions");
    expect(result).toHaveProperty("metadata");
    expect(result.questions).toHaveLength(20);
    expect(["rule_based_fallback", "ml_service"]).toContain(
      result.metadata.predictionSource
    );
    expect(result.metadata.totalQuestions).toBe(20);
  }, 15000);

  test("includes difficulty breakdown in metadata", async () => {
    const pool = createMockPool({ Easy: 30, Medium: 50, Hard: 20 });
    const result = await buildBalancedExam("student123", pool, [], 20);

    const { difficultyBreakdown } = result.metadata;
    expect(difficultyBreakdown).toHaveProperty("easy");
    expect(difficultyBreakdown).toHaveProperty("medium");
    expect(difficultyBreakdown).toHaveProperty("hard");

    const total =
      difficultyBreakdown.easy +
      difficultyBreakdown.medium +
      difficultyBreakdown.hard;
    expect(total).toBe(20);
  }, 15000);

  test("filters student history", async () => {
    const pool = createMockPool({ Easy: 10, Medium: 10, Hard: 10 });
    const history = [{ questionIds: pool.slice(0, 5).map((q) => q._id) }];

    const result = await buildBalancedExam("student123", pool, history, 10);

    const resultIds = result.questions.map((q) => q._id);
    for (let i = 0; i < 5; i++) {
      expect(resultIds).not.toContain(pool[i]._id);
    }
  });

  test("throws on empty pool", async () => {
    await expect(buildBalancedExam("student123", [], [])).rejects.toThrow(
      "Question pool is empty"
    );
  });

  test("throws on invalid totalQuestions", async () => {
    const pool = createMockPool({ Easy: 5, Medium: 5, Hard: 5 });
    await expect(
      buildBalancedExam("student123", pool, [], 0)
    ).rejects.toThrow("totalQuestions must be at least 1");
  });

  test("handles pool smaller than requested exam length", async () => {
    const pool = createMockPool({ Easy: 3, Medium: 3, Hard: 3 });
    const result = await buildBalancedExam("student123", pool, [], 40);

    // Should return whatever is available, not 40
    expect(result.questions.length).toBeLessThanOrEqual(9);
    expect(result.questions.length).toBeGreaterThan(0);
  });

  test("shuffles question order", async () => {
    const pool = createMockPool({ Easy: 20, Medium: 30, Hard: 15 });
    const results = [];

    for (let i = 0; i < 3; i++) {
      const result = await buildBalancedExam("student123", pool, [], 10);
      results.push(result.questions.map((q) => q._id).join(","));
    }

    // At least 2 distinct orderings out of 3 runs
    const unique = new Set(results);
    expect(unique.size).toBeGreaterThanOrEqual(2);
  }, 30000);
});
