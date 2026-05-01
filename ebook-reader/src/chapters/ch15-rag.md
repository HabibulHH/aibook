# Module 5: RAG — LLM-কে তোমার Knowledge দেওয়া

> "LLM অনেক কিছু জানে। কিন্তু তোমার business-এর কথা জানে না। RAG দিয়ে সেটা বদলে যায়।"

---

## Problem টা বোঝো

তোমার WhatsMonk customer — একটা clothing brand।

তাদের return policy আছে। Shipping rules আছে। Size guide আছে।

LLM এগুলো জানে না। Training-এ ছিল না।

তাহলে customer জিজ্ঞেস করলে "return policy কী?" — LLM কী বলবে?

হয় wrong answer দেবে, নয়তো বলবে "জানি না।"

দুটোই কাজের না।

**Solution: RAG।**

---

## RAG কী

RAG = Retrieval Augmented Generation

তিনটা word-এর মানে:

```
Retrieval   = relevant content খুঁজে আনো
Augmented   = সেটা দিয়ে LLM-কে augment করো
Generation  = LLM informed answer generate করে
```

Simple pipeline:

```
তোমার docs index করো (একবার)
         ↓
User question আসলে
         ↓
Similar content search করো
         ↓
LLM context-এ inject করো
         ↓
LLM সেটা দিয়ে answer করে
```

---

## Embedding কী — এটা না বুঝলে RAG বোঝা যাবে না

Text directly search করা যায় না similarity দিয়ে।

"return policy" আর "পণ্য ফেরত দেওয়ার নিয়ম" — same meaning, different words। Normal search মিস করবে।

Solution: text-কে numbers-এ convert করো।

```
"return policy"              → [0.23, 0.87, 0.12, 0.95...]
"পণ্য ফেরত দেওয়ার নিয়ম"   → [0.24, 0.85, 0.11, 0.94...]
```

Similar meaning → similar numbers। এটাই **embedding।**

Vector DB এই numbers store করে এবং similarity দিয়ে search করে।

```
তোমার question embed করো
         ↓
Vector DB-তে similar vectors খোঁজো
         ↓
Most similar docs বের হয়
```

---

## RAG Pipeline — Step by Step

**Phase 1: Indexing (একবার করো)**

```typescript
// তোমার docs load করো
const docs = [
  "Return policy: 7 দিনের মধ্যে return করা যাবে...",
  "Shipping: Dhaka-তে 2-3 দিন, outside 4-5 দিন...",
  "Size guide: S=36, M=38, L=40..."
]

// প্রতিটা chunk embed করো
for (const doc of docs) {
  const vector = await embed(doc)
  await vectorDB.upsert({ vector, content: doc })
}
```

**Phase 2: Retrieval + Generation (প্রতিটা request-এ)**

```typescript
async function ragQuery(userQuestion: string) {

  // 1. Question embed করো
  const questionVector = await embed(userQuestion)

  // 2. Similar docs খোঁজো
  const relevantDocs = await vectorDB.search(questionVector, { topK: 3 })

  // 3. LLM context-এ inject করো
  const messages = [
    {
      role: "system",
      content: `
        এই information দিয়ে answer দাও:
        ${relevantDocs.map(d => d.content).join('\n')}
      `
    },
    { role: "user", content: userQuestion }
  ]

  // 4. LLM answer করে
  return await llm.call({ messages })
}
```

---

## RAG কোথায় Fit করে — Memory Architecture-এ

Module 4-এ তিন ধরনের memory দেখেছিলাম।

```
Working Memory  = messages array    ← এই session
Episodic Memory = past summaries    ← আগের sessions
Semantic Memory = Vector DB         ← এটাই RAG
```

RAG হলো **Semantic Memory layer-এর implementation।**

তোমার static knowledge (docs, FAQs, policies) Vector DB-তে index করো। Relevant টুকু retrieve করে LLM-এ inject করো।

---

## RAG vs Tool Call — Important Distinction

এখানে অনেকে confuse হয়।

```
RAG:
→ static/semi-static knowledge
→ আগে থেকে indexed
→ "এই topic সম্পর্কে কী জানি?"
→ Vector DB search
→ Examples: docs, FAQs, policies, textbook content

Tool Call:
→ dynamic/real-time data
→ on-demand fetch
→ "এই moment-এ কী সত্যি?"
→ API/DB call
→ Examples: order status, payment status, live inventory
```

---

## দুটো একসাথে — Complete Picture

Real production-এ দুটোই লাগে।

```
Customer: "আমার payment failed, refund policy কী?"

RAG retrieves (static):
→ "Refund policy: 3-5 business days, original payment method-এ..."

Tool fetches (dynamic):
→ check_payment_status(trx_id) → "Failed at bKash gateway"
→ check_refund_eligibility(order_id) → "Eligible"

LLM gets both:
→ Policy জানে (RAG)
→ এই specific case জানে (Tool)
→ Personalized, accurate answer দেয়
```

---

## Per-User Context — Vector DB-তে রাখবে?

এখানে একটা common mistake আছে।

সব কিছু Vector DB-তে রাখতে হবে না।

```
এই data কি semantic search করবো?
(meaning-based similarity দরকার?)

Yes → Vector DB
No  → Traditional DB
```

User-এর extracted facts:
```
"COD prefer করে"          → Vector DB ✓ (semantic search)
"Mirpur-এ থাকে"           → Vector DB ✓
```

User-এর raw history:
```
Last 10 orders              → PostgreSQL ✓ (simple query)
Conversation history        → PostgreSQL ✓
```

Raw conversation কখনো Vector DB-তে রেখো না। Too expensive, too slow।

---

## Chunking — এটা Underrated Topic

তুমি একটা 50 page document index করবে। পুরো document একটা vector?

না। Chunk করো।

```
খুব বড় chunk:
→ irrelevant content mix হয়
→ retrieval quality কমে

খুব ছোট chunk:
→ context হারায়
→ incomplete answer

Sweet spot: 200-500 tokens per chunk
+ overlap রাখো (50-100 tokens)
  যাতে boundary-তে context না হারায়
```

---

## Advanced RAG — Production-এ যা লাগে

Basic RAG দিয়ে শুরু করো। কিন্তু production-এ এগুলো লাগবে।

**Query Rewriting:**
```
User লিখলো: "return korbo kivabe"
Rewrite করো: "return policy and process"
Better retrieval হয়
```

**Reranking:**
```
Vector search 10টা result দিলো
Reranker model best 3টা select করে
Accuracy বাড়ে
```

**HyDE (Hypothetical Document Embedding):**
```
Question-এর hypothetical answer লেখো
সেটা embed করে search করো
Better semantic match হয়
```

---

## Brritto-তে RAG কেমন হবে

তোমার Brritto-র জন্য RAG knowledge base:

```
Vector DB-তে index করবে:
├── NCTB textbook content (class 6-12)
├── Topic explanations
├── Common misconceptions
├── Model answers for CQ
├── Marking rubrics
└── Past board questions

Per-student memory:
├── "Weak in refraction"           → Vector DB
├── "Strong in algebra"            → Vector DB
├── Attempt history                → PostgreSQL
└── Session summaries              → PostgreSQL
```

Student wrong answer দিলে:
```
RAG: ঐ topic-এর explanation retrieve করো
Tool: student-এর mistake pattern fetch করো
LLM: personalized explanation দেয়
```

---

## এক লাইনে Module 5

```
RAG  = তোমার static knowledge → embed → Vector DB
     = user question এলে → similar content retrieve
     = LLM context-এ inject → informed answer

RAG vs Tool:
  RAG  = কী জানি (static)
  Tool = এখন কী সত্যি (dynamic)
  
দুটো মিলে = Complete AI system
```

এখন তুমি জানো LLM কী, Tools কী, Agents কী, Memory কী, RAG কী।

এগুলো কীভাবে production-এ deploy করবে, scale করবে, maintain করবে?

এটাই পরের module — **Production।**
