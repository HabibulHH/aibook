# Module 7: Harness Engineering — Agent এর চারপাশে যা থাকে

> "Model is the engine. Harness is the car। তুমি engine চালাও না, গাড়ি চালাও।" — Mitchell Hashimoto, 2026

---

## আগের Module-গুলোর Recap

এই বইয়ে এখন পর্যন্ত যা শিখেছো:

```
Module 1: LLM         → Brain কী, কীভাবে predict করে
Module 2: Tools       → Brain-কে চোখ-হাত দেওয়া
Module 3: Agents      → Goal দিয়ে নিজে কাজ করানো
Module 4: Tool Calling→ Agentic loop code-এ কীভাবে চলে
Module 5: Context     → State কীভাবে maintain হয়
Module 6: RAG         → External knowledge inject করা
Module 7: Production  → Demo থেকে real world
```

কিন্তু একটা প্রশ্ন এখনো clear না।

**যখন তুমি `weather_agent.py` চালাও — সেটা কি একটা agent? নাকি অর্ধেক agent?**

Demo-র জন্য fine। কিন্তু Claude Code, Cursor, Devin — এদের ভেতরে কী আছে যেটা তোমার script-এ নেই?

উত্তর — **Harness।**

এই module-এ সেই missing piece টা ছিঁড়ে ছিঁড়ে দেখবো।

---

## নতুন একটা সমীকরণ

Mitchell Hashimoto (HashiCorp এর co-founder) 2026 সালে একটা শব্দ দিল যেটা পুরো industry-তে standard হয়ে গেছে:

```
Agent = Model + Harness
```

পাঁচ শব্দ। কিন্তু এই সমীকরণ-টাই এই module এর core।

```
Model    = LLM (Claude, GPT, Gemini, Llama)
           Stateless brain
           Token in, token out
           
Harness  = Model-এর চারপাশে সব কিছু
           Tools, memory, validation, safety,
           observability, retry, cost control
           
Agent    = এই দুইটা একসাথে যা produce করে
           Reliable, useful, production-ready system
```

**Important:** Model alone is **not** an agent। `client.messages.create()` কে agent বলো না। সেটা একটা API call। Harness যোগ করলে — তখন agent।

---

## কেন এই শব্দ এখন আসলো?

2024 সালে সবাই "AI agent" বলতো — কিন্তু practice-এ সবাই আলাদা জিনিস বুঝতো।

```
কেউ বলতো:    "Agent = LLM + tools"
কেউ বলতো:    "Agent = ReAct loop"
কেউ বলতো:    "Agent = autonomous task solver"
কেউ বলতো:    "Agent = chatbot এর next version"
```

এই vague definition-এর কারণে production-এ মানুষ বিপদে পড়লো।

OpenAI এর Codex team 2026-এ একটা confession দিল — তাদের একটা production application-এ ১০ লাখ লাইনের code, যার একটা লাইনও human লেখেনি। কিন্তু engineers-রা code লেখেননি — তারা **harness** লিখেছেন। Sam Altman নিজে বললো:

> "It's hard to overstate how critical the harness is. I no longer think of the harness and the model as these entirely separable things."

**Mindset shift:** Model টা হলো GPU-র কাজ। Harness তোমার কাজ। Software engineer হিসেবে তুমি harness build করো।

---

## Analogy 1 — Engine vs Car

```
Model       =  Engine
Harness     =  Steering wheel
              + Seatbelts
              + Dashboard
              + Brakes
              + GPS
              + Airbags
              + Fuel system

Agent       =  পুরো গাড়ি
```

Engine শক্তিশালী হলেই গাড়ি চালানো যায় না। Steering নেই — কোথায় যাবে জানে না। Brake নেই — থামতে পারে না। Dashboard নেই — কী হচ্ছে দেখতে পারো না।

LLM exactly that engine। Harness সেই গাড়ি।

---

## Analogy 2 — Harness = AI এর Fullstack

Backend developer হিসেবে তুমি fullstack এর concept জানো:

```
Frontend     →  User interact করে
Backend      →  Business logic
Database     →  Data persist
Infrastructure → Deploy, scale
```

Harness-এ exactly same structure — কিন্তু agent এর জন্য:

| Fullstack | Harness | কাজ |
|---|---|---|
| Frontend / UI | **Guides** (AGENTS.md, system prompt) | User intent → agent এ পৌঁছায় |
| Backend / API | **Sensors** (validators, evals) | Rules enforce করে |
| Database | **Context Layer** | History, certified data persist |
| Docker / K8s | **Agentic Loop** | চালু রাখে, scale করে |
| Nginx | **Permission System** | কে কী করতে পারবে |
| Datadog | **Observability** | কী হচ্ছে দেখে |

**একটা পার্থক্য:**

```
Fullstack এ:   Human click করে → frontend → backend → DB → ...
Harness এ:    Human → Guide → LLM → Tool → LLM → Tool → ...
                              ↑___________↑___________↑
                                LLM drives the loop
```

Fullstack এ human drives। Harness-এ LLM drives — human শুধু শুরু আর শেষে।

এই কারণেই Harness Engineering আলাদা একটা discipline হয়ে দাঁড়িয়েছে। Web app বানাতে fullstack শেখো। AI app বানাতে — harness শেখো।

---

## তিনটা Layer

Harness কী এটা বুঝলে। এখন harness-এর ভেতরে কী আছে?

মূলত তিনটা layer:

```
┌────────────────────────────────────────────┐
│                                            │
│   Layer 1: GUIDES                          │
│   (Agent কে কী করতে বলো)                  │
│                                            │
│   ─────── ↓ ──────────                     │
│                                            │
│   Layer 2: SENSORS                         │
│   (Agent কী করল সেটা check করো)           │
│                                            │
│   ─────── ↓ ──────────                     │
│                                            │
│   Layer 3: CONTEXT LAYER                   │
│   (Agent কে কী data দিচ্ছো)               │
│                                            │
└────────────────────────────────────────────┘
                 ↑
                 │
            MODEL (LLM)
```

পরের তিন module-এ প্রতিটা layer ছিঁড়ে দেখবো। আজকে শুধু overview।

---

## Layer 1 — Guides (Constitution)

Guides হলো agent-এর জন্য লেখা rule-book।

```
Guides include:
  • System prompt (প্রতি session-এ inject হয়)
  • AGENTS.md / CLAUDE.md / .cursorrules
  • Constraint documents (style guide, conventions)
  • Tool descriptions
  • Permission rules
```

PocketSchool example:

```markdown
# AGENTS.md

## Project: PocketSchool LMS

## Code conventions
- NestJS modules, not Express routes
- Prisma for DB (no raw SQL)
- বাংলা UI text DB থেকে আসবে, hardcode না

## Test rules
- Write tests before merging
- Integration tests hit real Postgres (not mocked)

## DO NOT
- Modify schema.prisma without `pnpm migrate:dev`
- Touch .env (production credentials)
- Use console.log (use Logger service)
```

**মূল কথা:** "তোমার coding standards follow করো" — এটা probabilistic। AGENTS.md-এ লেখা থাকলে agent সাধারণত follow করে। কিন্তু linter দিয়ে block করলে — সেটা deterministic।

Guides + enforcement = real harness। শুধু লিখে রাখলে fully harness না।

পরের module-এ (Module 8: Guides) এটা detail-এ যাবো।

---

## Layer 2 — Sensors (Immune System)

Sensors হলো — agent action নেওয়ার পরে check করার system।

চার ধরনের sensor:

```
1. Evals (computational)
   লিন্টার, type checker, unit test
   Pass / fail। Deterministic।
   
2. LLM-as-judge
   আরেকটা LLM output evaluate করে
   Quality, tone, correctness check
   
3. Validation loop
   Agent দাবি করে done — কিন্তু test না চললে
   "done" mark করতে দেয় না
   
4. Drift detector
   আগে কাজ করতো, এখন behavior পাল্টেছে?
   Model drift বা data drift catch করে
```

PocketSchool example:

```python
# Sensor: agent code লিখলে test না চলা পর্যন্ত done না
def post_code_change(agent_output):
    test_result = subprocess.run(["pnpm", "test"], capture_output=True)
    if test_result.returncode != 0:
        return {
            "status": "blocked",
            "feedback": test_result.stderr.decode(),
            "instruction": "Fix the failing tests before completing"
        }
    return {"status": "ok"}
```

**Stat যেটা চমকে দেবে:** 65% enterprise AI failure trace করে harness defect-এ — model defect না। যখন সবাই "GPT-5 আসলে আমার agent ঠিক হবে" ভাবে — আসলে তাদের sensor layer ভাঙা।

---

## Layer 3 — Context Layer (Ground Truth)

Context layer হলো — agent যে data দেখে decision নেয়।

```
Context layer-এ থাকে:
  • Conversation history
  • Tool results (recent)
  • Memory (long-term, persisted)
  • Certified data (verified, schema-checked)
  • Lineage info (এই data কোথা থেকে এলো)
```

**সবচেয়ে বিপজ্জনক failure:** Context Rot।

```
দৃশ্য:
  ৩ মাস আগে AGENTS.md-এ লেখা ছিল —
  "users table-এ email column আছে"

  গত সপ্তাহে migration হয়েছে —
  email → email_address (rename)

  আজকে agent SQL লিখলো:
    SELECT email FROM users
  
  Result: error? না।
  Result: NULL columns? হয়তো।
  Result: stale cache হিট? সম্ভব।
  
  Agent confidently wrong answer দিলো।
  No error। শুধু bad output।
```

এই কারণেই context layer-এ "certified data" এর concept এসেছে। প্রতি table-এ:
- Schema verified (auto-check)
- Quality rules passing
- Owner acknowledged
- Last certified date logged

Gartner বলেছে — context management এর কারণে 40%+ agentic project 2027-এর মধ্যে বাতিল হবে।

---

## পুরো Picture একসাথে

```
┌─────────────────────────────────────────────────┐
│  USER intent                                    │
│         ↓                                        │
│  ┌──────────────────────────────────────────┐  │
│  │  GUIDES (constitution)                    │  │
│  │  System prompt + AGENTS.md + rules        │  │
│  └──────────────────────────────────────────┘  │
│         ↓                                        │
│  ┌──────────────────────────────────────────┐  │
│  │  AGENTIC LOOP                             │  │
│  │                                            │  │
│  │   while not done:                          │  │
│  │     LLM call (with context)                │  │
│  │     ↓                                      │  │
│  │     Tool call → execute                   │  │
│  │     ↓                                      │  │
│  │     Append result to context              │  │
│  │     ↓                                      │  │
│  │     Repeat                                 │  │
│  └──────────────────────────────────────────┘  │
│         ↓                                        │
│  ┌──────────────────────────────────────────┐  │
│  │  SENSORS (immune system)                  │  │
│  │  Evals, validators, drift detectors       │  │
│  └──────────────────────────────────────────┘  │
│         ↑                                        │
│  ┌──────────────────────────────────────────┐  │
│  │  CONTEXT LAYER (ground truth)             │  │
│  │  History, memory, certified data          │  │
│  └──────────────────────────────────────────┘  │
│         ↓                                        │
│  RESPONSE → USER                                │
└─────────────────────────────────────────────────┘
```

এই picture-টা মাথায় রাখো। পরের কয়েকটা module এই layer-গুলো এক এক করে detail-এ explain করবে।

---

## Tool vs Tool Call vs Harness — ভুলে যাওয়ার আগে

অনেকে এই তিনটা গুলিয়ে ফেলে। Clear করে রাখি।

| Concept | মানে | Example |
|---|---|---|
| **Tool** | Agent-এর capability (function) | `read_file`, `send_email` |
| **Tool Call** | LLM যখন decide করে tool use করবে | Single API request-এ structured request |
| **Context Window** | LLM এর working memory (limited) | 200k tokens |
| **Harness** | এই সব কিছুকে wrap করা system | Tools + context manager + sensors + loop |

Simple analogy:

```
Tool          = হাতুড়ি
Tool Call     = হাতুড়ি দিয়ে একটা ঘা
Context Window= কাজের টেবিল (সীমিত জায়গা)
Harness       = পুরো workshop
                (টেবিল + হাতুড়ি + storage +
                 safety rules + কে কী করতে পারবে)
```

---

## Claude Code = Real Harness Engine

Theory বুঝলে। এখন একটা real implementation দেখি।

```
Claude Code
├── Model: Claude (Anthropic API)
└── Harness:
    ├── Guides
    │   ├── CLAUDE.md (project rules)
    │   ├── System prompt
    │   └── Permission policy
    │
    ├── Sensors
    │   ├── Hooks (PreToolUse, PostToolUse)
    │   ├── Output parser
    │   └── Test runner integration
    │
    ├── Context Layer
    │   ├── File reader
    │   ├── Memory system (persistent)
    │   ├── Compaction (Dream Consolidation)
    │   └── Conversation history
    │
    └── Agentic Loop
        ├── Tool registry (60+ tools)
        ├── Sub-agent spawning
        └── Permission gate
```

Claude Code নিজে কোনো "intelligent" না। সে শুধু harness — যেটা LLM-কে wrap করে useful বানায়।

**যা তুমি built করতে পারো:** এই pattern টাই। ছোট আকারে। এই book-এর বাকি module-গুলো সেই journey।

---

## কেন তুমি নিজে Harness বানাবে?

Claude Code, Cursor — এগুলো ব্যবহার করতে পারো। কিন্তু নিজে harness কেন বানাবে?

**তিনটা কারণ:**

```
1. Control
   Data তোমার server-এ থাকবে
   যেভাবে চাও configure করতে পারবে
   Cost optimize করতে পারবে

2. Domain-specific কাজ
   Claude Code general coding এর জন্য বানানো
   PocketSchool এর code শুধু PocketSchool agent বুঝবে
   বাংলায় কথা বলবে
   তোমার DB schema জানবে
   তোমার convention follow করবে

3. Product বানাবে
   Product-এ AI feature add করতে চাইলে
   Claude Code-কে user-এর হাতে দিতে পারবে না
   নিজের harness build করতে হবে
```

---

## এক লাইনে Module 7

```
Agent = Model + Harness

Model     = LLM (engine)
Harness   = চারপাশের সব
           = Guides + Sensors + Context Layer
             + Agentic loop + Permissions
             + Observability + Memory

পরের module-গুলোতে এই layer গুলো
এক এক করে build করবো।

Module 8:  Guides
Module 9:  Sensors
Module 10: Context Layer
Module 11: Mini Harness নিজে বানাও
Module 12+: Sub-agents, Hooks, MCP, ...
```

---

## পরের Module — Guides

Guides কী, কেন probabilistic আর deterministic instruction এর difference, AGENTS.md কীভাবে structure করো, system prompt কোথায় যায়, কীভাবে enforce করো — সব next module-এ।

আজকে যা মাথায় রাখো:

```
1. Model একা agent না। Model + Harness = Agent।
2. Harness-এর তিন layer: Guides, Sensors, Context।
3. 65% AI failure model-এর fault না — harness defect।
4. Claude Code এর success-এর বড় কারণ harness, model না।
5. তোমার product-এ AI আনতে হলে — নিজের harness build করতে হবে।
```

পরের module-এ Guides — agent-এর constitution। শুরু হবে AGENTS.md থেকে।

---

## 🎯 Quiz — Test Yourself

```quiz
[
  {
    "q": "Agent-এর সঠিক সমীকরণ কোনটা?",
    "options": [
      "A) Agent = LLM",
      "B) Agent = LLM + Tools",
      "C) Agent = Model + Harness",
      "D) Agent = ChatGPT-এর নতুন version"
    ],
    "correct": 2,
    "why": "Mitchell Hashimoto (2026) এই সমীকরণ standardize করেছেন। শুধু LLM = engine; harness যোগ করলে = পুরো car। `client.messages.create()` কে agent বলো না — সেটা একটা API call।"
  },
  {
    "q": "Harness-এর তিন layer কোনগুলো?",
    "options": [
      "A) Tools, Memory, API",
      "B) Guides, Sensors, Context Layer",
      "C) Frontend, Backend, Database",
      "D) System prompt, Tools, Tests"
    ],
    "correct": 1,
    "why": "Guides = constitution (agent-কে কী বলো)। Sensors = immune system (output check)। Context Layer = ground truth (কী data দিচ্ছ)।"
  },
  {
    "q": "Fullstack analogy-তে Sensors-এর equivalent কোনটা?",
    "options": [
      "A) Frontend / UI",
      "B) Backend / API (validators, business logic)",
      "C) Database",
      "D) Nginx (reverse proxy)"
    ],
    "correct": 1,
    "why": "Backend যেমন business rules enforce করে — sensors agent output validate করে। Frontend = Guides। Database = Context Layer। Nginx = Permission system।"
  },
  {
    "q": "Model alone কে কেন agent বলা যায় না?",
    "options": [
      "A) Model expensive তাই",
      "B) Tools, memory, validation, safety nai — শুধু stateless API call",
      "C) Model বাংলা বুঝে না",
      "D) Model GPU চায়"
    ],
    "correct": 1,
    "why": "Model = engine মাত্র। Harness ছাড়া reliability, safety, observability কিছুই নাই — সেটা agent না, just একটা API call।"
  },
  {
    "q": "65% enterprise AI failure-এর মূল কারণ কী?",
    "options": [
      "A) Model দুর্বল",
      "B) GPU costly",
      "C) Harness defect (context drift, schema misalignment, state degradation)",
      "D) User training-এর অভাব"
    ],
    "correct": 2,
    "why": "Industry data — model defect না, harness defect-ই 65% failure-এর root cause। GPT upgrade করে problem solve হয় না, harness ঠিক করতে হয়।"
  },
  {
    "q": "Harness Engineering আসলে কোন discipline-এর extension?",
    "options": [
      "A) Data Science",
      "B) ML Research",
      "C) Software Engineering",
      "D) Statistics"
    ],
    "correct": 2,
    "why": "Harness engineering web-এ fullstack-এর equivalent — শুধু agent-এর জন্য। Software engineer-এর দক্ষতা সরাসরি apply হয়।"
  }
]
```
