# Module 3: Agents — Chatbot থেকে Worker

> "Tool দিয়ে একটা কাজ হয়। Agent দিয়ে পুরো workflow হয়।"

---

## Chatbot আর Agent-এর পার্থক্য

তুমি একটা junior developer hire করলে।

**Chatbot mode:**
```
তুমি: "এই function টা লিখে দাও"
Junior: লিখে দিলো
তুমি: copy করলে, paste করলে, run করলে
তুমি: "error আসছে, fix করো"
Junior: fix করলো
তুমি: আবার copy-paste
```

তুমি driver। Junior শুধু suggestion দেয়।

**Agent mode:**
```
তুমি: "Brritto-তে topics collection slow কেন, fix করো"

Junior নিজেই:
  1. codebase read করলো
  2. query analyze করলো
  3. MongoDB index missing দেখলো
  4. fix লিখলো
  5. test লিখলো
  6. test run করলো
  7. pass হলো
  8. PR তৈরি করলো

তুমি: PR review করলে, merge করলে
```

তুমি reviewer। Agent কাজ করলো।

**এটাই পার্থক্য।**

---

## Agent কী — Book Definition

> **Agent = একটা AI system যে goal পায়, নিজে plan করে, tools use করে, loop-এ কাজ করে, goal achieve না হওয়া পর্যন্ত চালিয়ে যায়।**

Key word: **loop।**

Chatbot-এ loop নেই। তুমি বলো, সে বলো। Done।
Agent-এ loop আছে। Goal দাও, সে নিজেই চালায়।

---

## Agentic Loop — এটাই Core

```
┌─────────────────────────────────┐
│         Agentic Loop            │
│                                 │
│  Goal আসে                       │
│     ↓                           │
│  LLM thinks: কী করবো?          │
│     ↓                           │
│  Tool call করে                  │
│     ↓                           │
│  Result দেখে                    │
│     ↓                           │
│  LLM thinks: আর কী করতে হবে?  │
│     ↓                           │
│  আরেকটা tool call               │
│     ↓                           │
│  ... চলতে থাকে ...              │
│     ↓                           │
│  Goal achieved? → Done          │
└─────────────────────────────────┘
```

**Think → Act → Observe → Think → Act → Observe...**

এই loop-টাই agent-কে powerful করে।

---

## Agentic Coding — সবচেয়ে Trending Use Case

তুমি এখন Claude Code বা Cursor দেখছো। এগুলো exactly এটাই করে।

```
তুমি: "এই repo-তে payment integration add করো"

Agent:
  → repo structure read করে
  → existing payment code খোঁজে
  → missing parts identify করে
  → bKash integration লেখে
  → tests লেখে
  → test run করে
  → fail হলে নিজেই fix করে
  → pass হলে PR বানায়
```

তুমি শুধু final review করো।

**Traditional coding with AI:**
তুমি Driver। AI suggest করে।

**Agentic coding:**
তুমি Reviewer। AI করে।

---

## Agent Types — কোন কাজে কোনটা

**Coding Agents:**
```
Claude Code, Devin, Cursor Agent
→ codebase বোঝে, bug fix করে, PR বানায়
```

**Browser Agents:**
```
Operator, Browser Use
→ actual browser চালায়
→ form fill করে, data scrape করে
→ "LinkedIn থেকে 500 lead collect করো" — নিজেই করে
```

**Support Agents:**
```
Customer message আসে
→ intent বোঝে
→ DB/API check করে
→ reply করে
→ resolve না হলে human-এ escalate করে
```

**Data Agents:**
```
"Last month revenue কেন drop করলো?"
→ DB query করে
→ data analyze করে
→ chart বানায়
→ insight দেয়
```

**Voice Agents:**
```
Phone call receive করে
→ real-time কথা বলে
→ appointment book করে
→ support দেয়
```

---

## Multi-Agent Systems — Next Level

একটা agent সব করতে পারে না। Complex tasks-এ দরকার হয় specialist agents।

```
তুমি: "আমার YouTube channel-এর জন্য weekly content plan করো"

Orchestrator Agent:
  → Research Agent কে দাও: "trending topics find করো"
  → Research Agent: web search করে, data আনে
  
  → Writer Agent কে দাও: "5টা video idea লেখো"
  → Writer Agent: research data দিয়ে ideas লেখে
  
  → Reviewer Agent কে দাও: "ideas review করো"
  → Reviewer Agent: quality check করে
  
  → Final output: reviewed content plan
```

প্রতিটা agent specialized। Orchestrator coordinate করে।

এটাই **Multi-Agent System।**

---

## Production Reality — Agents কোথায় Fail করে

এখানে honest হওয়া দরকার।

Agents impressive। কিন্তু production-এ অনেক কিছু fail করে।

**কেন fail করে:**
```
1. Security/compliance handle করতে পারে না
2. Enterprise systems-এর সাথে integration ভাঙে
3. Governance নেই, audit trail নেই
4. Long-running tasks-এ context হারায়
5. Error recovery দুর্বল
```

Gartner predict করেছে — 40% agentic AI projects 2027-এর মধ্যে বাতিল হবে।

Model fail করে না। **Operationalization fail করে।**

তাই agent বানানোর সময় শুধু "কী করতে পারবে" না, "কোথায় থামবে" — এটাও design করতে হবে।

**Human in the loop** — agent কখন human-এর approval নেবে। এটা weakness না, strategy।

---

## এক লাইনে Module 3

```
Agent = LLM + Tools + Loop
      = Goal দাও, সে করে
      = তুমি Reviewer, সে Worker
      
Multi-Agent = Specialist agents + Orchestrator
            = Complex workflows handle করে
```

কিন্তু এই concept গুলো এখনো high-level। Code-এ আসলে কী হয়? Loop টা কোথায় শুরু, tool call request-এ কী আসে, তোমার code কী send করে?

এটাই পরের module — **Tool Calling Code-এ কীভাবে চলে।**
