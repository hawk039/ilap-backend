import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from app.evaluation.evaluator import evaluate_all

def generate_report():
    results = evaluate_all()
    
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    
    # Identify borderline cases (Risk analysis)
    # Borderline: Confidence between 0.25 and 0.35 (near the 0.3 threshold)
    borderline_cases = [
        r for r in results 
        if 0.25 <= r["confidence"] <= 0.35
    ]
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{(passed/total)*100:.1f}%" if total > 0 else "0%"
        },
        "failures": [r for r in results if not r["passed"]],
        "borderline_cases": borderline_cases,
        "full_results": results
    }
    
    # Save to JSON
    output_path = Path(__file__).parent / "evaluation_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
        
    # Print Report
    print("\n" + "="*30)
    print("ILAP EVALUATION REPORT")
    print("="*30)
    print(f"Total Cases: {total}")
    print(f"Passed:      {passed}")
    print(f"Failed:      {failed}")
    print("-" * 30)
    
    if failed > 0:
        print("\nFAILURES:")
        for f in report["failures"]:
            print(f"- {f['id']}: {f['reason']}")
    else:
        print("\nNo failures detected.")
        
    if borderline_cases:
        print("\n⚠️ RISKY BORDERLINE CASES (0.25 - 0.35):")
        for b in borderline_cases:
            print(f"- {b['id']}: Confidence {b['confidence']} (Expected Answer: {b['expected_answer']})")
            
    print(f"\nFull report saved to: {output_path}")

if __name__ == "__main__":
    generate_report()
