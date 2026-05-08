# Module 9: Sensors — Agent কী করল সেটা Check করো

> "Guide লেখা সহজ। Agent সেটা মানে কিনা — সেটা verify করার system দরকার। সেটাই Sensor।"

---

## আগের Module Recap

Module 8-এ শিখেছো — Guides agent-কে rules বলে।

```
"AGENTS.md এ লেখা — test চালাবে"
                 ↓
   Agent ৭০% সময় follow করে
                 ↓
   ৩০% সময়? Silent skip। Production-এ bug।
```

Guide **probabilistic**। তোমার dependency probability of compliance এ। Critical জিনিসে এটা যথেষ্ট না।

আজকে — **Sensor layer** যেটা guide-কে enforced করে। Probabilistic থেকে deterministic-এ shift।

---

## Sensor কী

Sensor হলো — agent action নেওয়ার পরে (বা আগে) automatic check।

```
Agent          →  output / action
                  ↓
Sensor         →  check করে
                  ↓
   PASS  →  proceed
   FAIL  →  block, alert, retry, log
```

**Software dev analogy:**

```
Code review        =  Manual sensor (slow, expert)
Linter             =  Computational sensor (fast, deterministic)
CI pipeline        =  Sensor chain (multiple checks)
Production monitor =  Drift sensor (long-running)
```

Agent-এর জন্যও same pattern।

---

## চার রকম Sensor

```
1. Computational Eval
   লিন্টার, type checker, unit test
   Deterministic, fast, no LLM cost

2. LLM-as-Judge
   আরেকটা LLM output evaluate করে
   Subjective quality, tone, correctness

3. Validation Loop
   Agent দাবি করে done — কিন্তু check fail করলে
   "done" mark করতে দেয় না, retry forces

4. Drift Detector
   Production-এ behavior change detect করে
   Model drift, data drift catch করে
```

এক এক করে দেখি।

---

## Sensor 1 — Computational Eval

সবচেয়ে সহজ। Software dev tools দিয়েই হয়।

**PocketSchool example:** agent code লিখলো — output check করো।

```python
def check_agent_code(file_path):
    checks = []
    
    # 1. Syntax (Python parse করতে পারে?)
    try:
        compile(open(file_path).read(), file_path, "exec")
        checks.append(("syntax", True))
    except SyntaxError as e:
        checks.append(("syntax", False, str(e)))
    
    # 2. Lint (PocketSchool style)
    lint = subprocess.run(
        ["pnpm", "lint", file_path],
        capture_output=True
    )
    checks.append(("lint", lint.returncode == 0, lint.stderr.decode()))
    
    # 3. Type check (TypeScript)
    types = subprocess.run(
        ["pnpm", "tsc", "--noEmit"],
        capture_output=True
    )
    checks.append(("types", types.returncode == 0, types.stderr.decode()))
    
    # 4. Tests
    tests = subprocess.run(
        ["pnpm", "test", file_path],
        capture_output=True
    )
    checks.append(("tests", tests.returncode == 0, tests.stdout.decode()))
    
    return checks
```

**Properties:**

```
✅ Deterministic     — পাস বা fail
✅ Fast              — milliseconds
✅ Cheap             — কোনো LLM cost না
✅ Auditable         — log দেখে কেন fail বুঝতে পারো
❌ Limited scope     — শুধু যা rule-এ লেখা যায়
```

**যা catch করে:**

- Syntax error
- Type mismatch
- Style violation
- Test failure
- Schema violation (JSON, DB)
- Security issue (secret detection, SAST)

**যা catch করে না:**

- "Code টা কি readable?"
- "Function naming meaningful?"
- "User experience ভালো?"
- Subjective quality

এই gap-টাই LLM-as-judge fill করে।

---

## Sensor 2 — LLM-as-Judge

আরেকটা LLM (সাধারণত cheaper model) প্রথম agent-এর output review করে।

```python
def llm_judge_review(code: str, requirements: str) -> dict:
    judge_prompt = f"""
    You are reviewing code written by another AI agent.
    
    Requirements:
    {requirements}
    
    Code:
    {code}
    
    Score (1-5) on:
    1. Correctness
    2. Readability
    3. PocketSchool conventions
    4. Test coverage
    
    Return JSON:
    {{
      "scores": {{"correctness": N, ...}},
      "issues": ["..."],
      "verdict": "approve" | "revise"
    }}
    """
    
    response = client.messages.create(
        model="claude-haiku-4-5",  # ← cheap model judge হিসেবে
        max_tokens=1024,
        messages=[{"role": "user", "content": judge_prompt}]
    )
    
    return json.loads(response.content[0].text)
```

**যখন use করবে:**

```
✅ Subjective quality assessment
✅ Tone, voice check (বাংলা translation কোয়ালিটি)
✅ "Did agent answer the actual question?"
✅ Hallucination detection (claim verification)

❌ Yes/No facts (computational eval ভাল)
❌ High-throughput pipeline (cost বেড়ে যাবে)
```

**Tip:** Judge model ছোট রাখো (Haiku/Sonnet) — main agent Opus হলেও।

**Bias warning:** Judge LLM author LLM-কে favor করতে পারে (same family হলে)। Production-এ different vendor judge use করা হয় কখনো।

---

## Sensor 3 — Validation Loop

এটা Sensor 1 আর Sensor 2 কে **agentic loop-এ inject** করার pattern।

**Problem:** Agent তিনটা tool call-এর পরে বলে "Done!" — কিন্তু test fail করেছে। সে নিজে check করেনি।

**Solution:** Agent কে done বলতে দিও না — sensor pass না হলে।

```python
def run_with_validation(user_input):
    messages = [{"role": "user", "content": user_input}]
    
    while True:
        response = llm_call(messages)
        
        # Tool call হলে normal handle
        if response.stop_reason == "tool_use":
            execute_and_append(response, messages)
            continue
        
        # Agent বললো done — কিন্তু validate করো
        validation = run_sensors()
        
        if validation["pass"]:
            return response.text  # ← OK
        
        # Sensor fail — agent কে error feedback দিয়ে retry forced
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": (
                f"Validation failed:\n{validation['errors']}\n"
                f"Fix the issues and try again."
            )
        })
        # Loop continue
```

**Effect:**

```
Agent: "I added the quiz feature, done!"
Sensor: "test/quiz.spec.ts failed — TypeError on line 42"
Agent: "Sorry, let me check that..."
        → reads quiz.spec.ts
        → fixes the bug
        → "Done now"
Sensor: "All tests pass ✓"
```

**Hallucinated completion** এই pattern-এ blocked হয়। Agent বলতেই পারবে না "done" যতক্ষণ sensor ok না।

**Max retry limit add করো:**

```python
retries = 0
MAX_RETRIES = 5

while retries < MAX_RETRIES:
    # ... loop
    if not validation["pass"]:
        retries += 1
        continue

if retries >= MAX_RETRIES:
    return "Could not complete task. Last errors: ..."
```

Infinite retry যাতে না হয়।

---

## Sensor 4 — Drift Detector

Production-এ deploy করার পরের sensor।

**Problem:**
```
Day 1:   Agent works perfectly
Day 30:  Same code, behavior subtly off
Day 60:  Customers complain
Day 90:  ক্ষতি হয়ে গেছে
```

কেন drift হলো?

```
Model drift:
  - API provider new model deploy করল
  - Same model name, different weights
  - Behavior subtly changed

Data drift:
  - Input patterns পাল্টেছে
  - User base বদলেছে
  - Edge cases-এর distribution change
```

**Detection pattern:**

```python
# Golden dataset
GOLDEN_TESTS = [
    {
        "input": "Find quizzes for class 5 math",
        "expected_tools": ["search_quizzes"],
        "expected_output_contains": ["math", "class 5"]
    },
    # ... 100 more
]

def daily_drift_check():
    failures = []
    for test in GOLDEN_TESTS:
        result = run_agent(test["input"])
        if not matches(result, test):
            failures.append(test)
    
    if len(failures) > THRESHOLD:
        alert("Drift detected!", failures)
```

প্রতিদিন/সপ্তাহে golden test চালাও। Failure rate বাড়লে — drift।

**Industry data যেটা চমকে দিবে:**

```
65% enterprise AI failure trace করে
harness defect-এ — model defect না।

Top reasons:
  1. Context drift
  2. Schema misalignment  
  3. State degradation
```

মানে — model upgrade করে problem solve হবে না। Sensor ঠিক থাকলে, problem আগে catch হবে।

---

## Sensor কোথায় বসে — Architecture

```
                ┌─────────────┐
   Input ──────►│  PRE-SENSOR │  ← Schema check, permission
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │   AGENT     │
                │   (LLM +    │
                │   tools)    │
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │ POST-SENSOR │  ← Eval, judge
                └──────┬──────┘
                       │
                       ▼ pass
                    Output
                       │
                ┌──────▼──────┐
                │   DRIFT     │  ← Continuous monitor
                │   DETECTOR  │
                └─────────────┘
```

**Pre-sensor:** Agent-এর কাছে input পৌঁছানোর আগে check।
- Input format valid?
- User permission আছে?
- Rate limit?

**Post-sensor:** Agent output produce করার পরে check।
- Output schema valid?
- Hallucination?
- Test pass?

**Drift detector:** Background job — daily/weekly।
- Golden test pass rate
- Latency change
- Cost per task change

---

## Hook vs Sensor — পার্থক্য

কোডের দিক থেকে hook আর sensor কাছাকাছি। কিন্তু concept ভিন্ন।

| | Hook | Sensor |
|---|---|---|
| When | Lifecycle event-এ trigger | Validation step হিসেবে |
| Purpose | Side effect (log, format) | Pass/fail decision |
| Blocking | Sometimes | Always (pass না হলে block) |
| Example | "PostToolUse: log this" | "PostCode: tests must pass" |

Module 13-এ hooks-এর details আসবে। আপাতত — overlap থাকলেও separate concept।

---

## PocketSchool Sensor Stack

Real example: PocketSchool-এ একটা coding agent deploy করতে চাও। Sensor stack কেমন হবে?

```yaml
pre_sensors:
  - schema_check          # input format
  - rate_limit            # 10 req/min/user
  - permission            # only paid users

agent:
  model: claude-opus-4-7
  tools: [read_file, write_file, run_command, search]

post_sensors:
  - lint                  # pnpm lint
  - typecheck             # tsc --noEmit
  - unit_tests            # affected tests only
  - llm_judge             # readability score >3
  - secret_scan           # no .env exposed

validation_loop:
  max_retries: 5
  retry_on: [lint_fail, test_fail, secret_found]

drift:
  golden_tests: 50
  schedule: daily
  alert_threshold: 5%   # failure rate
```

প্রতিটা sensor একটা specific risk address করছে।

---

## ভুল যা সবাই করে

```
১. শুধু LLM-as-judge use করা
   → Computational eval সস্তা, deterministic
   → Judge subjective জিনিসে rakho

২. Sensor fail-এ silent log
   → Block করো, retry forced করো
   → Silent log = agent learns nothing

৩. সব sensor আগে চালানো
   → Cheap → Expensive order
   → Lint আগে, then test, then LLM judge

৪. Golden dataset stale
   → 6 মাস আগের golden test
   → New behavior detect করে না
   → Quarter-এ refresh

৫. Sensor-কে blackbox রাখা
   → Agent বুঝে না কেন fail
   → Detailed error message essential
```

---

## Cost Calculation

Sensor stack expensive হতে পারে। Budget-aware হও।

```
Per task cost:
  Lint           ~$0.00 (CPU time only)
  Typecheck      ~$0.00
  Unit test      ~$0.00 (CPU)
  LLM judge      ~$0.01-0.05 (Haiku)
  Drift check    daily one-time

Total post-sensor: ~$0.02 per task (mostly judge)
```

**Tip:** LLM judge শুধু critical paths-এ। Internal/dev features এ skip।

---

## Connection to Other Modules

```
Module 8 (Guides)
  → এখানে যা rule লিখলে, sensor enforce করে

Module 10 (Context)
  → Sensor ground truth চেক করে context-এর বিরুদ্ধে

Module 11 (Mini Harness)
  → Sensor layer code-এ bolt করবো

Module 13 (Hooks)
  → Implementation detail (PreToolUse, PostToolUse)
```

---

## Harness Coverage — Unsolved Problem

একটা open question industry-তে:

```
"Sensor never fires —
 মানে quality high?
 নাকি detection inadequate?"
```

কেউ সমাধান করতে পারেনি। **Harness coverage metric** — software code coverage-এর equivalent — এখনো standardize হয়নি।

```
Software:
  Line coverage    → কতটুকু code execute হলো test-এ
  Branch coverage  → সব path-এ test আছে?

Harness:
  Sensor coverage  → কতটা failure mode catch করতে পারে?
  উত্তর: ???
```

এটাই **harness debt** — পরে Module 19 (Harness Debt)-এ আরো details।

---

## এক লাইনে Module 9

```
Sensor      = Agent action validate করার system
Form 4 টা   = Computational + LLM judge
              + Validation loop + Drift detector

Mindset:
  Pre-sensor   = action আগে block
  Post-sensor  = action পরে validate
  Drift        = production-এ ongoing monitor

Coverage rule:
  Critical path → multiple sensor
  Cheap sensor আগে, expensive পরে
```

পরের module — **Context Layer** — agent কী data দেখে decision নিচ্ছে। সবচেয়ে underbuilt layer। 65% AI failure এই layer-এ।

---

## 🎯 Quiz — Test Yourself

```quiz
[
  {
    "q": "Sensor-এর ৪ type কোনগুলো?",
    "options": [
      "A) Read, Write, Execute, Search",
      "B) Computational eval, LLM-as-judge, Validation loop, Drift detector",
      "C) Linter, Test, Build, Deploy",
      "D) System prompt, AGENTS.md, Tool, Permission"
    ],
    "correct": 1,
    "why": "Computational eval (deterministic), LLM-as-judge (subjective), validation loop (force retry), drift detector (production monitor) — চারটাই complementary, একসাথে কাজ করে।"
  },
  {
    "q": "Computational eval-এর strength কোনটা?",
    "options": [
      "A) Subjective quality measure করে",
      "B) Deterministic, fast, cheap (no LLM cost)",
      "C) Hallucination detect করে",
      "D) Tone evaluate করে"
    ],
    "correct": 1,
    "why": "Linter, type checker, test — deterministic pass/fail, milliseconds-এ run, কোনো LLM cost নেই। Subjective জিনিসে limited — সেখানে LLM-as-judge use করো।"
  },
  {
    "q": "Hallucinated completion কী?",
    "options": [
      "A) Agent ভুল code লিখে",
      "B) Agent token শেষ করে fail করে",
      "C) Agent দাবি করে done — কিন্তু আসলে test fail বা কাজ অসম্পূর্ণ",
      "D) Agent infinite loop-এ আটকে যায়"
    ],
    "correct": 2,
    "why": "Agent বলে 'Done!' কিন্তু sensor check করলে দেখা যায় test fail বা task incomplete। Validation loop এই pattern block করে — sensor pass না হলে done mark করতে দেয় না।"
  },
  {
    "q": "LLM-as-judge কখন best use case?",
    "options": [
      "A) Yes/No facts check",
      "B) Schema validation",
      "C) Subjective quality (readability, tone, বাংলা translation কোয়ালিটি)",
      "D) Test execution"
    ],
    "correct": 2,
    "why": "Subjective জিনিস — code readability, tone, translation quality — যেগুলো simple rule-এ ধরা যায় না। Yes/No facts-এ computational eval সস্তা এবং deterministic।"
  },
  {
    "q": "Hook vs Sensor — মূল পার্থক্য কী?",
    "options": [
      "A) Hook = Python, Sensor = JavaScript",
      "B) Hook = lifecycle event side effect, Sensor = pass/fail decision (block ক্ষমতা)",
      "C) Hook = production-only, Sensor = dev-only",
      "D) কোনো পার্থক্য নাই — synonymous"
    ],
    "correct": 1,
    "why": "Hook lifecycle event-এ trigger হয় (log, format, side effect)। Sensor pass/fail decide করে — fail হলে block করে। Code-এ কাছাকাছি, concept ভিন্ন।"
  },
  {
    "q": "Drift detector কোন pattern catch করে?",
    "options": [
      "A) Lint error",
      "B) Test failure",
      "C) Day 1 কাজ করতো, Day 30-এ subtly behavior পাল্টেছে (model বা data)",
      "D) Syntax error"
    ],
    "correct": 2,
    "why": "Production-এ ongoing monitor। Model drift (provider new weights) বা data drift (input distribution change) detect করে — golden test daily চালিয়ে failure rate track করে।"
  }
]
```
