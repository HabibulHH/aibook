# Module 8: Guides — Agent-কে কী করতে বলবে

> "Guide হলো agent-এর constitution। এর বাইরে যাওয়া যাবে না। কিন্তু constitution আর law-এর মধ্যে পার্থক্য আছে।"

---

## আগের Module Recap

Module 7-এ শিখেছো — **Agent = Model + Harness**। Harness-এর তিন layer:

```
Guides    →  Agent কে কী বলো (constitution)
Sensors   →  কী করল check করো  (immune system)
Context   →  কী data দিচ্ছ      (ground truth)
```

আজকে প্রথম layer — **Guides** — এ deep dive।

---

## Guide আসলে কী

Guide হলো — agent-কে **আগেই** বলে দেওয়া rules, conventions, এবং goals।

```
Tool         = Agent এর হাত-পা
Context      = Agent এর চোখের সামনে যা আছে
Guide        = Agent এর মাথায় যা bake করা আছে
                (instruction set, identity, rules)
```

প্রতি conversation এ guide automatically inject হয়। User তা type করে না। Developer একবার লিখে রাখে।

---

## Guide-এর চার রকম form

```
1. System prompt
   API call-এ system="..." parameter
   প্রতি request এ যায়

2. AGENTS.md / CLAUDE.md / .cursorrules
   Repo-র root-এ markdown file
   Harness automatically পড়ে context এ যোগ করে

3. Tool descriptions
   Tool definition-এর description field
   কখন কোন tool use করতে হবে এই hint

4. Permission rules
   কোন tool কোথায় allow, কোথায় ask, কোথায় deny
```

এই চারটাই guide। Form ভিন্ন, intent এক — agent-কে boundary দাও।

---

## Form 1 — System Prompt

সবচেয়ে basic form। `weather_agent.py`-এ তুমি দেখোনি, কারণ default ছিল।

```python
response = client.messages.create(
    model="claude-opus-4-7",
    system="""
        You are PocketSchool's coding assistant.
        - বাংলা UI text DB থেকে আসবে, hardcode না
        - NestJS modules use করবে, raw Express না
        - প্রতি code change এর পরে test চালাবে
        - destructive command (rm, drop) আগে confirm চাইবে
    """,
    tools=tools,
    messages=messages,
)
```

`system` parameter হলো **persistent identity**। Agent এই rule গুলো প্রতি turn এ "মনে রাখে" (কারণ প্রতিবার পাঠানো হয়)।

**System prompt vs User prompt:**

| | System | User |
|---|---|---|
| কে লেখে | Developer | End user |
| কখন যায় | প্রতি call-এ inject | Session এর part |
| কে দেখে | User দেখে না | User typed |
| Override | User overide করতে পারে না (ideally) | প্রতি turn-এ change |

**System prompt-এ কী রাখো:**

```
✅ Identity (তুমি কে — coding assistant, support bot, ...)
✅ Tone (formal/informal, বাংলা/English)
✅ Constraints (কী করবে না)
✅ Tool selection hints (কখন কোন tool)
✅ Output format (JSON/markdown/plain)

❌ Dynamic data (user info, current date) → এটা context এ যাবে
❌ Long examples (token waste) → tool description এ যাবে
```

---

## Form 2 — AGENTS.md (এবং Friends)

System prompt API call-এর জন্য। কিন্তু Claude Code, Cursor, Codex — এদের সবাইকে same instruction দিতে হলে?

**একটাই file:** `AGENTS.md`।

2025 সালের August এ OpenAI, Google, Cursor, Anthropic — এদের collaborate-এ একটা open standard তৈরি হলো। Repo-র root-এ `AGENTS.md` রাখলে যেকোনো agent harness সেটা পড়ে context এ inject করে।

PocketSchool এর `AGENTS.md` example:

```markdown
# PocketSchool LMS — Agent Guide

## Stack
- Backend: NestJS + Prisma + Postgres
- Frontend: Next.js (app router) + TanStack Query
- Mobile: React Native (Expo)

## Code conventions

### Backend
- Module structure: `src/modules/<feature>/`
- Controller-Service-Repository pattern
- Validation: class-validator DTOs
- Error: throw HttpException, never return error object

### Database
- Migration: `pnpm prisma migrate dev`
- Schema source of truth: `schema.prisma`
- Seed file: `prisma/seed.ts`

### Frontend
- বাংলা UI text DB থেকে আসবে, source code-এ hardcode না
- API call: TanStack Query, no fetch in component
- State: Zustand for client, TanStack for server

## Test rules
- Unit test: `*.spec.ts` co-located
- Integration: `test/` folder, real Postgres
- e2e: Playwright

## Commands
- Dev: `pnpm dev`
- Test: `pnpm test`
- Migrate: `pnpm migrate:dev`
- Lint: `pnpm lint`

## DO NOT
- Touch `.env` (production credentials)
- Modify `schema.prisma` without running migration
- Use `console.log` (use Logger service)
- Add new dependency without checking package.json size
```

এই file টা **machine-readable**। Agent এটা পড়ে — knows your project।

**পার্থক্য Vendor-by-Vendor:**

| Vendor | File name | Behavior |
|---|---|---|
| OpenAI Codex | `AGENTS.md` | Auto-loaded |
| Cursor | `.cursorrules` | Auto-loaded |
| Claude Code | `CLAUDE.md` (also reads `AGENTS.md`) | Auto-loaded |
| Aider | `.aider.conf.yml` + conventions | Manual reference |
| Devin | `playbook.md` (similar) | Auto-loaded |

**Best practice:** একটা `AGENTS.md` রাখো — সবাই পড়ে। Vendor-specific file থাকলেও সেটা `AGENTS.md` পড়ে।

---

## Form 3 — Tool Description (Hidden Guide)

Module 4-এ দেখেছিলে tool definition:

```python
{
    "name": "get_weather",
    "description": (
        "Get the current weather for a given location. "
        "Returns temperature (°C), wind speed (km/h), and a short condition summary."
    ),
    "input_schema": {...}
}
```

এই `description` field — এটাও **guide**। LLM এটা পড়ে decide করে কখন এই tool call করবে।

**খারাপ description:**
```python
"description": "Weather tool"
```

LLM হয়তো call করবে, হয়তো করবে না। Confused।

**ভাল description:**
```python
"description": (
    "Get current weather for a location. "
    "Use when user asks about weather, temperature, "
    "rain, or planning outdoor activity. "
    "Returns temperature (°C), wind, condition. "
    "Don't use for historical or forecast — only current."
)
```

কখন use করবে, কী return করে, কখন **use করবে না** — সব লেখা।

**Rule of thumb:**

```
description = mini guide for that one tool

Include:
  • কাজ কী
  • কখন use করবে
  • কখন use করবে না (anti-pattern)
  • Return value-র shape
```

---

## Form 4 — Permission Rules

Guide শুধু text না। Configuration ও guide।

Claude Code-এর `settings.json` example:

```json
{
  "permissions": {
    "allow": [
      "Bash(pnpm test:*)",
      "Bash(pnpm lint)",
      "Read",
      "Edit"
    ],
    "ask": [
      "Bash(pnpm migrate:*)"
    ],
    "deny": [
      "Bash(rm -rf:*)",
      "Bash(git push --force:*)",
      "Edit(.env*)"
    ]
  }
}
```

```
allow → silently execute
ask   → user confirm করে
deny  → তোমার harness block করবে
```

এটাও guide — কিন্তু **enforced** guide। এর পর সরাসরি Sensor territory শুরু।

---

## Probabilistic vs Deterministic — সবচেয়ে Important Distinction

এই section টা **মন দিয়ে পড়ো।** Harness Engineering এর core mindset shift এখানেই।

```
"AGENTS.md-এ লেখা — test চালাবে"
                ↓
   Probabilistic compliance: ~70%

"Hook configured — post-edit hook test runs"
                ↓
   Deterministic enforcement: 100%
```

Guide তে শুধু লিখলে — agent **মান্য করবে probability of N**। Test চালানো, secret check, format keep — সব probability-driven।

**যেটা must (security, data integrity) — সেটা deterministic হতে হবে।**

কীভাবে?

```
১. Pre-tool hook   →  tool execute হওয়ার আগে check
২. Post-tool hook  →  execute হওয়ার পরে validate
৩. Linter / type checker  →  agent run করার বাইরে
৪. Test gate       →  test fail হলে done mark করতে দিও না
৫. Permission deny →  deny rule = hard block
```

পরের module (Sensors) এই deterministic enforcement-এ deep dive।

**এক লাইনে:** Guide তে যা তুমি বলবে, agent করার চেষ্টা করবে। কিন্তু critical জিনিস **system দিয়ে enforce করো**, শুধু text-এ ভরসা না।

---

## কেন Constraint থাকলে Agent বেশি Productive

একটা counter-intuitive observation:

```
Free agent     →  any approach try করে
                  Token waste, wrong path, slow

Constrained    →  narrow option space
                  Faster convergence, predictable
```

**Example:**

Free PocketSchool agent:
```
User: "Add a quiz feature"

Agent thinks:
  - Maybe new microservice?
  - Maybe in monolith?
  - Express or NestJS?
  - SQL or NoSQL?
  - REST or GraphQL?

Tries 3 approaches → 5000 tokens wasted
```

Constrained agent (with AGENTS.md):
```
User: "Add a quiz feature"

Agent reads AGENTS.md:
  ✓ NestJS module structure
  ✓ Prisma + Postgres
  ✓ Controller-Service pattern

Goes straight to:
  src/modules/quiz/
    quiz.module.ts
    quiz.controller.ts
    quiz.service.ts
  → 1500 tokens, done
```

Constraint = focus। Less freedom = more productivity।

---

## Anatomy — একটা ভাল AGENTS.md কী Include করে

PocketSchool AGENTS.md থেকে pattern বের করি।

```
1. Project context (১-২ লাইন)
   "PocketSchool LMS for Bangladeshi students"

2. Stack (bullet list)
   Backend, frontend, mobile, DB

3. Conventions per layer
   Backend rules
   Database rules
   Frontend rules

4. Test rules (concrete)
   Where, how, what kind

5. Commands (copy-pasteable)
   pnpm dev, pnpm test, ...

6. DO NOT (explicit anti-patterns)
   Most important section
   Security, data integrity items
```

**Length:** 50-200 lines। বেশি লিখলে — agent পড়বে, কিন্তু signal-to-noise কমে যাবে।

**Update frequency:** Quarter-এ একবার review। Stale guide = harmful guide।

---

## ভুল যা সবাই করে

```
১. সব কিছু system prompt-এ ঠেসা
   → Token খরচ বাড়ে, attention diffuse হয়
   → Solution: AGENTS.md এ stable rules,
     system prompt-এ identity + dynamic info

২. AGENTS.md never updated
   → Schema পাল্টালো, AGENTS.md নাই
   → Agent stale info দিয়ে বিভ্রান্ত হয়
   → Solution: Schema PR template-এ
     "AGENTS.md update?" checkbox

৩. শুধু do, কোনো don't নাই
   → Agent jane কী করতে হবে,
     জানে না কী করতে নেই
   → Solution: "DO NOT" section essential

৪. Guide-এ enforcement নাই
   → "Test চালাবে" — কিন্তু hook নাই
   → Probabilistic compliance only
   → Solution: critical rules → sensor

৫. Vendor-specific guide লেখা
   → শুধু .cursorrules, AGENTS.md নাই
   → Tool change করলে guide harano
   → Solution: AGENTS.md primary,
     others reference সেটা
```

---

## PocketSchool এর জন্য Practical Setup

Step 1 — Repo root এ `AGENTS.md` create করো।

Step 2 — চারটা section minimum:

```markdown
# PocketSchool — Agent Guide

## Stack
[stack list]

## Conventions
[code rules]

## Commands
[bash commands]

## DO NOT
[anti-patterns]
```

Step 3 — System prompt-এ identity:

```python
SYSTEM_PROMPT = """
You are PocketSchool's coding assistant.
Read AGENTS.md before any task.
Communicate in Bengali, code in English.
Always run tests after code changes.
"""
```

Step 4 — Tool description audit। প্রতিটা tool-এ "when to use" এবং "when not to" লেখো।

Step 5 — Permission config — destructive command deny।

এই পাঁচ step পুরো guide layer cover করে।

---

## Connection to Other Modules

```
Module 7 (Intro)
  → এই module Layer 1 detail

Module 9 (Sensors)
  → Guide-এর critical rules sensor-এ enforce

Module 10 (Context)
  → Dynamic data context-এ যাবে, guide-এ না

Module 11 (Mini Harness)
  → Guide layer code-এ implement করবো
```

---

## এক লাইনে Module 8

```
Guide      = Agent-কে আগেই বলা rules
Form 4 টা  = System prompt + AGENTS.md
              + Tool description + Permissions

Mindset:
  Probabilistic guide  → agent চেষ্টা করে
  Deterministic enforce → system বাধ্য করে

Critical জিনিস always deterministic।
Style preference probabilistic ok।
```

পরের module — **Sensors**। Guide তে লিখলে agent follow করল কি না কীভাবে check করবে। Evals, validation loops, drift detection — সব next module-এ।

---

## 🎯 Quiz — Test Yourself

```quiz
[
  {
    "q": "Guide-এর ৪টা form কোনগুলো?",
    "options": [
      "A) System prompt, AGENTS.md, Tool description, Permission rules",
      "B) Frontend, Backend, Database, API",
      "C) Linter, Type checker, Test, Drift detector",
      "D) Read, Write, Search, Execute"
    ],
    "correct": 0,
    "why": "Guide ৪ form-এ আসে — system prompt (API call-এ inject), AGENTS.md (repo-তে file), tool description (tool definition-এ hint), permission rules (allow/ask/deny config)।"
  },
  {
    "q": "নিচের কোনটা probabilistic compliance?",
    "options": [
      "A) Linter দিয়ে block করা",
      "B) Pre-commit hook test চালানো",
      "C) AGENTS.md-এ লেখা 'test চালাবে'",
      "D) Permission deny rule"
    ],
    "correct": 2,
    "why": "AGENTS.md-এ শুধু লিখলে agent ~70% follow করে — probability of compliance। Linter, hook, permission deny — সব deterministic enforcement।"
  },
  {
    "q": "AGENTS.md কেন vendor-neutral standard?",
    "options": [
      "A) GitHub এটা invent করেছে",
      "B) OpenAI, Google, Cursor, Anthropic একসাথে standard বানিয়েছে (2025)",
      "C) Microsoft এটা push করেছে",
      "D) এটা শুধু Cursor-এ কাজ করে"
    ],
    "correct": 1,
    "why": "August 2025-এ multiple vendor (OpenAI, Google, Cursor, Anthropic) collaborate করে AGENTS.md কে open standard হিসেবে চালু করেছে — যেকোনো harness পড়তে পারে।"
  },
  {
    "q": "System prompt-এ কোনটা রাখা ঠিক না?",
    "options": [
      "A) Agent identity (তুমি কে)",
      "B) Constraints (কী করবে না)",
      "C) Tool selection hints",
      "D) Current user-এর name এবং dynamic data"
    ],
    "correct": 3,
    "why": "Dynamic data context-এ যাবে, system prompt-এ না। System prompt stable identity + rules রাখে — প্রতি call-এ পাঠানো হয় বলে token waste।"
  },
  {
    "q": "Constraint বেশি দিলে agent কী হয়?",
    "options": [
      "A) Slow হয় — কম option try করতে পারে",
      "B) Confused হয়ে যায়",
      "C) Faster converge করে — narrow option space",
      "D) Tool call করতে পারে না"
    ],
    "correct": 2,
    "why": "Counter-intuitive কিন্তু true — constraint = focus। Free agent token waste করে exploring, constrained agent direct path-এ যায়।"
  },
  {
    "q": "AGENTS.md-এ সবচেয়ে important section কোনটা?",
    "options": [
      "A) Stack list",
      "B) Conventions",
      "C) Commands",
      "D) DO NOT (anti-patterns explicitly বলা)"
    ],
    "correct": 3,
    "why": "DO NOT section-এ security এবং data integrity-এর critical rules থাকে। শুধু 'কী করবে' বললে agent জানে না কী করতে নেই — explicit anti-pattern essential।"
  }
]
```
