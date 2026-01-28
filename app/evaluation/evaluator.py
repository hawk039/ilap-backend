import yaml
import sys
from pathlib import Path

# Add project root to sys.path to allow imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from app.services.answer_service import get_answer
from app.responses.refusals import NO_LAW_FOUND, NON_LEGAL_QUERY

def load_test_cases():
    path = Path(__file__).parent / "test_cases.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)

def evaluate_case(case):
    query = case["query"]
    expected = case["expected"]
    case_id = case["id"]
    
    # Call the system
    response = get_answer(query)
    
    passed = True
    reason = "Passed"
    
    # Check 1: Should Answer vs Refusal
    if expected["should_answer"]:
        if response.confidence < 0.3: # Using the threshold from answer_service
            passed = False
            reason = f"Expected answer, but got refusal (Confidence: {response.confidence})"
        elif response.answer in [NO_LAW_FOUND, NON_LEGAL_QUERY]:
                passed = False
                reason = f"Expected answer, but got refusal message: {response.answer}"
        
        # Check 2: Section Correctness (if specified)
        if passed and "section" in expected:
            found_section = False
            for citation in response.citations:
                # Simple substring match for robustness
                if expected["section"] in citation.section:
                    found_section = True
                    break
            if not found_section:
                passed = False
                citations_str = ", ".join([c.section for c in response.citations])
                reason = f"Expected section {expected['section']}, found: {citations_str}"

    else: # Should NOT answer
        if response.confidence >= 0.3:
                passed = False
                reason = f"Expected refusal, but got answer (Confidence: {response.confidence})"
        
    return {
        "id": case_id,
        "query": query,
        "passed": passed,
        "reason": reason,
        "confidence": response.confidence,
        "expected_answer": expected["should_answer"]
    }

def evaluate_all():
    test_cases = load_test_cases()
    results = []
    
    print(f"Running evaluation on {len(test_cases)} cases...\n")
    
    for case in test_cases:
        result = evaluate_case(case)
        results.append(result)
        
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"Testing: {result['id']} -> {status} | {result['reason']}")

    return results

if __name__ == "__main__":
    evaluate_all()
