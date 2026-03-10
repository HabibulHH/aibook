# Module 4: Context আর Memory — State কে Manage করে?

> "LLM stateless। কিন্তু তোমার agent-এর state দরকার। এই tension টা solve করাই এই module-এর কাজ।"

---

## Problem টা বোঝো

Agent loop চলছে। Multiple tool calls হচ্ছে।

```
Tool Call 1: get_order_status(1234)
Tool Call 2: check_bkash_payment(trx_id)
Tool Call 3: get_delivery_estimate(location)
```

LLM কীভাবে জানবে Call 1-এর result Call 3-এ relevant?

Module 1-এ বলেছিলাম — LLM stateless। প্রতিটা call fresh।

তাহলে?

**Answer: LLM জানে না। তুমি জানাও।**

Context hold করে তুমি — LLM না।

---

## Messages Array — তুমিই Memory

তুমি একটা growing messages array maintain করো। প্রতিটা LLM call-এ পুরোটা পাঠাও।

```typescript
const messages = [
  // conversation শুরু
  { role: "user", content: "আমার order 1234 status কী?" },

  // LLM বললো tool call দরকার
  { role: "assistant", tool_calls: [
    { name: "get_order_status", args: { id: "1234" } }
  ]},

  // tool result inject করলে
  { role: "tool", content: "Shipped via Pathao, out for delivery" },

  // LLM আবার call — এখন আগের সব context দেখছে
  { role: "assistant", tool_calls: [
    { name: "check_bkash_payment", args: { trx_id: "TRX999" } }
  ]},

  // আরেকটা tool result
  { role: "tool", content: "Payment confirmed, 850 BDT" },

  // LLM এখন সব দেখে final reply করে
  { role: "assistant", content: "আপনার payment confirmed এবং order shipped!" }
]
```

প্রতিটা LLM call-এ **পুরো array** পাঠাচ্ছো। এটাই context।

---

## Agent Loop-এ এটা কেমন দেখায়

```typescript
async function agentLoop(userMessage: string) {

  const messages = [
    { role: "user", content: userMessage }
  ]

  while (true) {

    // পুরো messages array পাঠাও
    const response = await llm.call({
      messages: messages,   // ← সব history
      tools: availableTools
    })

    if (response.tool_calls) {

      // assistant message রাখো
      messages.push({
        role: "assistant",
        tool_calls: response.tool_calls
      })

      // প্রতিটা tool execute করো, result inject করো
      for (const call of response.tool_calls) {
        const result = await executeTool(call.name, call.args)
        messages.push({
          role: "tool",
          content: JSON.stringify(result)
        })
      }

      // loop continue — LLM আবার call হবে updated context দিয়ে

    } else {
      return response.content  // final answer
    }
  }
}
```

Loop চলে যতক্ষণ LLM বলে "আর tool দরকার নেই।"

---

## 3 ধরনের Memory

এতক্ষণ দেখলাম working memory। কিন্তু memory আসলে তিন ধরনের।

```
┌──────────────────────────────────────────┐
│                                          │
│  Working Memory    = messages array      │
│  এই conversation    RAM এর মতো          │
│                                          │
│  Episodic Memory   = past conversations  │
│  আগের কী হয়েছে     HDD এর মতো          │
│                                          │
│  Semantic Memory   = facts/knowledge     │
│  general জ্ঞান      Vector DB            │
│                                          │
└──────────────────────────────────────────┘
```

**Working Memory** — এই session-এর messages। Redis-এ রাখো, session শেষে expire।

**Episodic Memory** — গতকাল user কী বলেছিলো। PostgreSQL-এ। Conversation শেষে summary save করো।

**Semantic Memory** — "এই user COD prefer করে।" Vector DB-তে। Semantic search করে retrieve করো।

---

## Large Scale-এ Problem

Messages array বাড়তে থাকে।

```
10 tool calls  = 10 messages
100 turns      = 100 messages
→ context window overflow
→ cost বাড়ে (প্রতিটা token = money)
→ latency বাড়ে
```

দুটো technique দিয়ে solve করো।

**Sliding Window:**
```
Max 20 messages রাখো
নতুন আসলে সবচেয়ে পুরনো drop করো

[msg1, msg2...msg20]
নতুন আসলে:
[msg2, msg3...msg21]  ← msg1 gone
```

**Summarization:**
```typescript
if (messages.length > 50) {
  const summary = await llm.summarize(messages.slice(0, 40))
  messages = [
    { role: "system", content: `Previous context: ${summary}` },
    ...messages.slice(40)  // শেষ 10টা full রাখো
  ]
}
```

---

## Selective Context Loading — Advanced

সব memory একসাথে load করো না। Relevant টুকু load করো।

```typescript
async function buildContext(userMessage, userId) {

  // 1. Recent messages (last 10) — Working memory
  const recent = await redis.get(sessionId)

  // 2. Relevant past facts — Semantic search
  const relevant = await vectorDB.search(
    await embed(userMessage),
    { topK: 3, filter: { userId } }
  )

  // 3. User profile — Structured facts
  const profile = await db.getUserProfile(userId)

  return [
    { role: "system", content: `
        User: ${profile}
        Relevant history: ${relevant}
    `},
    ...recent,
    { role: "user", content: userMessage }
  ]
}
```

---

## Production Tools — নিজে বানাতে হবে না

এই সব নিজে implement করা complex। Production-এ existing tools use করো।

**Mem0:**
```
User memory layer
Cross-session memory
Automatic fact extraction
Production-ready
```

**Zep:**
```
Conversation memory store
Automatic summarization
Temporal awareness (কখন কী হয়েছিলো)
```

**LangMem:**
```
LangChain-এর memory module
Long-term memory for agents
Relevance-based retrieval
```

এগুলো basically তোমার হয়ে memory management করে।

---

## Real Numbers — কখন কী করবে

```
Messages < 20     → full array pass করো
Messages 20-50    → sliding window + summarize old
Messages > 50     → semantic retrieval only
Multi-session     → episodic + semantic memory
Cost বেশি        → aggressive summarization
```

---

## এক লাইনে Module 4

```
Context    = Messages Array (তুমি maintain করো)
Memory     = Working + Episodic + Semantic (তিন layer)
LLM        = stateless, প্রতিবার fresh

তুমি array build করো।
LLM শুধু reason করে।
তুমিই memory।
```

কিন্তু এখন পর্যন্ত dynamic data নিয়ে কথা হলো।

Static knowledge কী? তোমার product docs, FAQs, policies — এগুলো কীভাবে LLM-কে দেবে?

এটাই পরের module — **RAG।**
