import json
import time
from semantic_explainer import SemanticExplainer

def run_validation_suite():
    print("========================================================================")
    print("      AVIATION METEOROLOGY - SEMANTIC ML BENCHMARK & VALIDATION         ")
    print("========================================================================")
    
    start_time = time.time()
    explainer = SemanticExplainer()
    init_duration = time.time() - start_time
    print(f"[+] Semantic ML Pipeline ready in {init_duration:.2f}s\n")

    test_scenarios = [
        {
            "category": "Atmosphere & Structure (Synonym / Rephrased)",
            "query": "What is the lowest layer of Earth's atmosphere?",
            "expected_keyword": "Troposphere"
        },
        {
            "category": "Thermal Structure & Inversion",
            "query": "At 80 km altitude in mesopause, what is the temperature in Kelvin?",
            "expected_keyword": "173"
        },
        {
            "category": "Altimetry & Pressure Settings",
            "query": "When flying from High to Low pressure area, does altimeter over read or under read?",
            "expected_keyword": "Over"
        },
        {
            "category": "Winds & Coriolis Force",
            "query": "Where is the Coriolis force maximum on the globe?",
            "expected_keyword": "Poles"
        },
        {
            "category": "Clouds & Optical Phenomena",
            "query": "Which cloud causes the Halo optical ring around the sun or moon?",
            "expected_keyword": "CS"
        },
        {
            "category": "Ice Accretion & Aviation Hazards",
            "query": "Which type of ice on airframe is glassy and hard to break?",
            "expected_keyword": "Glaze"
        },
        {
            "category": "Thunderstorm & Convective Hazards",
            "query": "What is the concentrated downdraught from TS of diameter less than 4 km called?",
            "expected_keyword": "Microburst"
        }
    ]

    passed = 0
    total = len(test_scenarios)

    for i, test in enumerate(test_scenarios, 1):
        print("-" * 75)
        print(f"Test #{i} [{test['category']}]")
        print(f"Query: \"{test['query']}\"")
        
        t0 = time.time()
        result = explainer.explain(test['query'])
        latency = (time.time() - t0) * 1000
        
        top = result["top_concept"]
        matched_q = top["question"]
        ans = top["correct_answer"]
        expl = top["scientific_explanation"]
        conf = top["confidence_level"]
        score = top["neural_relevance_score"]

        # Check semantic alignment
        is_hit = (
            test["expected_keyword"].lower() in ans.lower() or
            test["expected_keyword"].lower() in expl.lower() or
            test["expected_keyword"].lower() in matched_q.lower()
        )

        if is_hit:
            passed += 1
            status_str = "PASSED [OK]"
        else:
            status_str = "FLAGGED"

        print(f"Status:        {status_str} (Latency: {latency:.1f}ms)")
        print(f"Matched Q:     {matched_q}")
        print(f"Answer:        ({top['correct_option']}) {ans}")
        print(f"Neural Conf:   {conf} (Logit: {score})")
        print(f"Explanation:   {expl[:120]}...")

    print("========================================================================")
    print(f"Semantic Validation Score: {passed}/{total} ({passed/total * 100:.1f}%) Passed")
    print("========================================================================")

if __name__ == "__main__":
    run_validation_suite()
