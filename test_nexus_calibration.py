import requests
import json
import time
import sys
from datetime import datetime

# Fix Unicode output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_URL = "https://nexus-api-wud4.onrender.com/api/nexus/chat"

TESTS = {
    "greetings": ["Hi", "Hello", "What's this?", "Who are you?", "Hey"],
    "discovery": [
        "What do you do?",
        "How can you help my business?",
        "What is Barrios A2I?",
        "I run an e-commerce store",
        "I own a law firm",
        "I have a restaurant",
        "I'm a real estate agent",
        "I run a marketing agency"
    ],
    "objections": [
        "This sounds too expensive",
        "AI is too complicated for my team",
        "We tried AI before and it didn't work",
        "I'm worried AI will replace my employees",
        "My business is too unique for AI",
        "My customers hate talking to bots",
        "I don't have time for new technology",
        "How do I know this isn't just hype?"
    ],
    "roi_proof": [
        "What ROI can I expect?",
        "Do you have case studies?",
        "Show me proof this works",
        "How long until I see results?",
        "What results have other clients gotten?"
    ],
    "pricing": [
        "How much does this cost?",
        "What's your pricing?",
        "That's more than I expected",
        "What do I get for $759?",
        "What's the monthly fee for?"
    ],
    "closing": [
        "I want to learn more",
        "Can we schedule a call?",
        "I'm ready to get started",
        "What are the next steps?",
        "I want to talk to a human"
    ],
    "hard_questions": [
        "Can you build me a website?",
        "What makes you different from ChatGPT?",
        "Is my data secure?",
        "What if the AI gives wrong answers?",
        "Why not just use free AI tools?"
    ]
}

def test_message(msg, session_id):
    try:
        r = requests.post(BASE_URL,
            json={"message": msg, "session_id": session_id},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        full_response = ""
        for line in r.text.split('\n'):
            if line.startswith('data:'):
                try:
                    data = json.loads(line[5:].strip())
                    if data.get('type') == 'delta':
                        full_response += data.get('text', '')
                except:
                    pass
        return full_response.strip() or r.text[:500]
    except Exception as e:
        return f"ERROR: {e}"

def grade_response(msg, response):
    issues = []
    grade = "A"

    sentences = response.count('.') + response.count('?') + response.count('!')
    if sentences > 6:
        issues.append("Too long (>6 sentences)")
        grade = "B"
    if sentences < 2:
        issues.append("Too short")
        grade = "B"

    if response.count('\n-') > 2 or response.count('\n•') > 2 or response.count('**') > 4:
        issues.append("Too listy/formatted")
        grade = "C"

    if any(word in msg.lower() for word in ['expensive', 'cost', 'roi', 'results', 'proof', 'case study']):
        if not any(char.isdigit() for char in response):
            issues.append("Missing statistics")
            grade = "B" if grade == "A" else grade

    if not response.rstrip().endswith('?'):
        issues.append("No qualifying question at end")

    if len(response) > 800:
        issues.append("Response too long (>800 chars)")
        grade = "C" if grade in ["A", "B"] else grade

    return grade, issues

def run_calibration():
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("NEXUS BRAIN CALIBRATION - FAST API TEST")
    print("=" * 60)

    total = sum(len(msgs) for msgs in TESTS.values())
    current = 0

    for category, messages in TESTS.items():
        print(f"\n### {category.upper()} ###")
        session_id = f"cal-{category}-{timestamp}"

        for msg in messages:
            current += 1
            print(f"\n[{current}/{total}] Testing: {msg}")
            response = test_message(msg, session_id)
            grade, issues = grade_response(msg, response)

            results.append({
                "category": category,
                "message": msg,
                "response": response,
                "grade": grade,
                "issues": issues
            })

            print(f"Grade: {grade} | Length: {len(response)} chars")
            print(f"Response preview: {response[:120]}...")
            if issues:
                print(f"Issues: {', '.join(issues)}")

            time.sleep(1.5)

    # Generate report
    grades = [r['grade'] for r in results]
    all_issues = []
    for r in results:
        all_issues.extend(r['issues'])

    report = f"""# Nexus Brain Calibration Report
Generated: {datetime.now()}

## Summary
- **Total tests:** {len(results)}
- **A grades:** {grades.count('A')} ({100*grades.count('A')//len(results)}%)
- **B grades:** {grades.count('B')}
- **C grades:** {grades.count('C')}
- **F grades:** {grades.count('F')}
- **Overall Score:** {10 * grades.count('A') // len(results)}/10

## Top Issues to Fix
"""
    for issue in sorted(set(all_issues), key=lambda x: all_issues.count(x), reverse=True):
        report += f"- {issue}: {all_issues.count(issue)} occurrences\n"

    report += "\n## Results by Category\n"
    for category in TESTS.keys():
        cat_results = [r for r in results if r['category'] == category]
        cat_grades = [r['grade'] for r in cat_results]
        report += f"\n### {category.upper()}\n"
        report += f"Score: {cat_grades.count('A')}/{len(cat_grades)} A's\n\n"

        for r in cat_results:
            emoji = {"A": "✅", "B": "⚠️", "C": "❌", "F": "🚫"}.get(r['grade'], "?")
            report += f"{emoji} **{r['message']}** (Grade: {r['grade']})\n"
            report += f"> {r['response'][:300]}{'...' if len(r['response']) > 300 else ''}\n"
            if r['issues']:
                report += f"> Issues: {', '.join(r['issues'])}\n"
            report += "\n"

    with open("nexus_calibration_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print(f"A: {grades.count('A')} | B: {grades.count('B')} | C: {grades.count('C')}")
    print("Report saved: nexus_calibration_report.md")
    print("=" * 60)

    return results

if __name__ == "__main__":
    run_calibration()
