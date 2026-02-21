# Chapter 8: Production Reality — Licensing, Cost, আর Real World

> "Model free তো, তাহলে ধরে নিচ্ছি যেকোনো কিছুতে ব্যবহার করতে পারি? আর cost এর ব্যাপারটা কী?"

---

## Open Source মানেই Free না

এইটা সবচেয়ে common misconception। তুমি Hugging Face থেকে model download করলে, fine-tune করলে — মানেই এই না তুমি সেটা দিয়ে যেকোনো কিছু করতে পারো।

ML model এর licensing একটা minefield। npm এ বেশিরভাগ package MIT/Apache license — practically করো যা খুশি। কিন্তু ML model এ?

**LLaMA (Meta)** — Meta এর custom license আছে। Monthly active users 700 million এর বেশি হলে আলাদা agreement লাগে। Commercial use allowed কিন্তু conditions সহ।

**কিছু model "open weights"** — মানে তুমি use করতে পারো, inference করতে পারো, কিন্তু commercially retrain করতে পারো না।

**Revenue threshold restrictions** — কিছু model এ বলা আছে certain revenue এর উপরে গেলে license আলাদা।

Rule: **SaaS product বানানোর আগে model এর license card পড়ো।** Hugging Face এ প্রতিটা model এর page এ license mention থাকে।

---

## Self-Host নাকি API Call?

তোমার product এ AI feature add করতে চাও। দুইটা option:

### Option 1: API Call (OpenAI, Anthropic, Google)

```python
# OpenAI API
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "..."}]
)
```

**Pros:** Setup নাই, scale automatic, maintenance নাই, শুরু করতে ৫ মিনিট লাগে।

**Cons:** Per-token cost, data privacy concern, rate limits, provider dependency।

**Cost:** GPT-4o → ~$2.50/1M input tokens, ~$10/1M output tokens। ছোট scale এ cheap।

### Option 2: Self-Host (Own Server)

```bash
# vLLM দিয়ে model serve
vllm serve meta-llama/Llama-3-8B --port 8000
```

**Pros:** Full control, no per-token cost, data privacy, no rate limits।

**Cons:** GPU instance cost (~$1-3/hr), maintenance, scaling তোমাকে করতে হবে, DevOps overhead।

### কোনটা কখন?

| Scenario | Recommendation |
|----------|---------------|
| MVP / prototype | API call — fastest to market |
| < 1000 requests/day | API call — cheaper |
| > 10,000 requests/day | Self-host evaluate করো |
| Data privacy critical | Self-host |
| Indie hacker / startup | API call দিয়ে শুরু করো |
| Bangladesh market (low ARPU) | API cost carefully calculate করো |

**Break-even point:** সাধারণত হাজার হাজার requests/hour না হলে API সস্তা। তোমার WhatsMonk, BhaloHotels এর early stage এ definitely API call ভালো।

---

## Fine-tuning vs RAG vs Prompting

তিনটা way আছে AI model কে তোমার specific task এ ভালো করানোর। এগুলো জানা production decision এর জন্য critical:

### Prompting — সবচেয়ে সস্তা, প্রথমে এটা Try করো

```python
system_prompt = """You are a Bengali customer support agent for BhaloHotels. 
Always respond in Bengali. Be polite and helpful.
Hotel policies: check-in 2pm, check-out 12pm, free cancellation 24hr before."""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "আমি কি booking cancel করতে পারবো?"}
    ]
)
```

Cost: শুধু API call cost। কোনো training cost নাই।
Effort: System prompt লেখা — ঘণ্টায় হয়ে যায়।
Limitation: Complex domain knowledge এ কম effective।

### RAG (Retrieval Augmented Generation) — তোমার Data + LLM

এইটা তোমার জন্য সবচেয়ে relevant। ধরো PocketSchool এ students AI কে course content নিয়ে প্রশ্ন করবে। তুমি পুরো course content fine-tune করতে পারো না (expensive)। বরং:

**Step 1:** Course content কে chunks এ ভাগ করো
**Step 2:** প্রতিটা chunk এর embedding বানাও
**Step 3:** Embeddings vector database এ রাখো (pgvector — তোমার PostgreSQL এ)
**Step 4:** User প্রশ্ন করলে — প্রশ্নের embedding match করো database এর সাথে
**Step 5:** Relevant chunks LLM কে context হিসেবে দাও
**Step 6:** LLM সেই context থেকে answer দেয়

```python
# Simplified RAG flow
question_embedding = embed(user_question)
relevant_docs = vector_db.similarity_search(question_embedding, top_k=5)

response = llm.chat(
    system="Answer based on the provided context only.",
    context=relevant_docs,
    question=user_question
)
```

Cost: Embedding generation + API call। Fine-tuning এর চেয়ে অনেক সস্তা।
Effort: Vector DB setup + chunking strategy।
Best for: Company docs, course content, FAQ, knowledge base।

**pgvector** — PostgreSQL extension। তোমার existing Postgres stack এ vector search add করে। আলাদা database লাগে না!

```sql
CREATE EXTENSION vector;
CREATE TABLE documents (id serial, content text, embedding vector(1536));
SELECT * FROM documents ORDER BY embedding <-> query_embedding LIMIT 5;
```

### Fine-tuning — Expensive কিন্তু Best Domain Result

Model কে তোমার data দিয়ে re-train করো। Chapter 7 এ details আছে।

Cost: GPU cost (cloud বা own)। Hours থেকে days লাগতে পারে।
Effort: Data preparation + training + evaluation।
Best for: Very specific domain, unique tone/style, production quality।

### Rule of Thumb

```
প্রথমে Prompting try করো
  → কাজ হলে? ✅ Done
  → কাজ না হলে? ↓

RAG try করো
  → কাজ হলে? ✅ Done (বেশিরভাগ ক্ষেত্রে এখানেই solve হয়)
  → কাজ না হলে? ↓

Fine-tuning করো
  → Last resort, but best quality
```

---

## Model Serving — Production এ কিভাবে চলে?

তুমি model locally চালালে — ঠিক আছে। কিন্তু production এ ১০০০ user একসাথে request করলে? শুধু model load করে call করলেই হবে না।

Production serving এ যা handle করতে হয়:

**Batching** — অনেক request একসাথে group করে process করা। একটা একটা করে process করলে slow।

**KV-Cache Management** — LLM conversation এ previous context cache করে রাখা। না করলে প্রতিটা message এ পুরো conversation re-process করতে হবে।

**Auto-scaling** — request বাড়লে instance বাড়ানো, কমলে কমানো।

**Queue Management** — request overflow হলে queue করা।

**Health Checks** — GPU OOM (Out of Memory) detect করা, auto-restart।

Tools:

**vLLM** — সবচেয়ে popular LLM serving tool। PagedAttention দিয়ে GPU memory efficiently use করে।

**TGI (Text Generation Inference)** — Hugging Face এর solution। Docker image দিয়ে deploy।

**Triton Inference Server** — NVIDIA এর। Multi-model serving support করে।

ভাবো — এটা basically **nginx/load balancing কিন্তু GPU workload এর জন্য।** তোমার backend experience এখানে directly কাজে আসবে।

---

## Ecosystem Change — কিভাবে Cope করবে?

Backend engineering এ PostgreSQL fundamentals গত ১০ বছরে তেমন বদলায়নি। কিন্তু ML ecosystem? ৩ মাসে state-of-the-art model outdated হয়ে যায়।

তোমার strategy:

**Model-agnostic architecture বানাও।** AI layer কে abstract করো তোমার application এ। একটা interface বানাও:

```typescript
// AI Service Interface
interface AIService {
  chat(prompt: string): Promise<string>;
  embed(text: string): Promise<number[]>;
  classify(text: string): Promise<string>;
}

// OpenAI implementation
class OpenAIService implements AIService { ... }

// Anthropic implementation  
class AnthropicService implements AIService { ... }

// Self-hosted implementation
class LocalLLMService implements AIService { ... }
```

এখন model provider বদলালে শুধু implementation বদলাও, পুরো app rewrite না।

**এইটাই সবচেয়ে important architectural decision তোমার SaaS products এর জন্য।**

---

## তোমার Products এ সবচেয়ে Practical AI Integration কোনটা?

সব theory শেষে, practical answer:

**Embeddings + pgvector = তোমার first AI feature।**

কারণ:
- তোমার PostgreSQL stack এ fit করে
- Semantic search instantly better হয়
- RAG pipeline এর foundation
- Setup time কম
- Cost কম

**WhatsMonk** → Customer message intent classification (embedding similarity দিয়ে)
**BhaloHotels** → Hotel search "beach hotel near Cox's Bazar" (semantic search)
**PocketSchool** → Course content QA (RAG with pgvector)
**Edokan BD** → Product search improvement (semantic matching)

---

## এই Chapter এ কী শিখলে?

Open source মানেই free commercial use না — license পড়ো। API call vs self-host — early stage এ API, scale এ self-host evaluate করো। Prompting → RAG → Fine-tuning — এই order এ try করো। Production serving নিজস্ব engineering challenge — vLLM, TGI use করো। Model-agnostic architecture বানাও — AI layer abstract রাখো। আর তোমার products এ pgvector + embeddings দিয়ে শুরু করো।

পরের chapter — শেষ chapter — পুরো hands-on। Model download, run, API serve, fine-tune — সব একসাথে করবো।
