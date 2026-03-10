# Module 2: Tools — Blind Brain-কে চোখ দেওয়া

> "LLM একা অনেক কিছু জানে। কিন্তু কিছু করতে পারে না। Tools দিয়ে সেটা বদলে যায়।"

---

## আগের module পড়ে তোমার মাথায় যা এসেছে

চলো honest হই।

Module 1 পড়ে তুমি excited হয়েছো — LLM কত কিছু জানে, কত ভালো reason করে। তারপর মাথায় এসেছে: "তাহলে এটা দিয়ে আমার product বানাই।"

তুমি বানালে একটা customer support bot। WhatsApp-এ customer লিখলো:

> "আমার order 1234 কোথায়?"

LLM উত্তর দিলো:

> "আপনার order সম্পর্কে জানতে আপনার courier-এর website visit করুন অথবা customer care-এ call করুন।"

তুমি frustrated। এই জন্য LLM শিখলাম?

**Problem টা LLM-এর না। Problem টা তোমার expectation-এ।**

Module 1-এ একটা জিনিস clearly বলা হয়েছিল — LLM **stateless**, **text in text out only**, এবং তোমার database-এর কিছুই সে জানে না।

তাহলে সে order status কোথা থেকে জানবে?

এই gap টা পূরণ করে **Tools।**

কিন্তু এখানে একটা twist আছে। Tools মানে নতুন কোনো technology না। তুমি developer হিসেবে যা জানো — functions লেখো, API call করো — সেটাই। শুধু পার্থক্য একটাই: **কে call করার decision নেয়।**

আগে তুমি decide করতে। এখন LLM decide করে।

---

## Problem টা আগে বোঝো

তাহলে WhatsApp customer জিজ্ঞেস করলো "আমার order কোথায়?" — LLM কী বলবে?

```
LLM একা:
"আমি জানি না আপনার order কোথায়। 
 আপনার tracking number দিয়ে courier website চেক করুন।"
```

এটা কোনো কাজের answer না।

তোমার database-এ order আছে। Pathao-এর API-তে tracking আছে। কিন্তু LLM সেটা access করতে পারছে না।

এই problem solve করে **Tools।**

---

## Tool কী — Book Definition

> **Tool = একটা function যেটা LLM নিজে execute করতে পারে না, কিন্তু তোমার system-কে বলতে পারে "এটা execute করো।"**

মানে হলো —

LLM বলে: "আমার এই data দরকার।"
তোমার code সেটা fetch করে।
Result LLM-এ আসে।
LLM সেটা দিয়ে answer করে।

LLM নিজে কিছু করে না। সে শুধু **decide করে** কোন tool call করতে হবে।

---

## Function Call-এর সাথে পার্থক্য কী?

তুমি developer — তুমি জানো function call কী।

```typescript
// Normal function call — তুমি decide করো
if (needOrderStatus) {
  getOrderStatus(orderId)
}

// Tool — LLM decide করে
// তুমি শুধু tool define করো
// LLM নিজেই বলে "এখন getOrderStatus call করো"
```

**পার্থক্য একটাই — কে call করার decision নেয়।**

Normal function: তুমি।
Tool: LLM।

OpenAI এটাকে শুরুতে "Function Calling" বলতো। পরে সবাই "Tool Use" বলা শুরু করলো। Same concept, different name।

---

## Tool কীভাবে Define করে

```typescript
const tools = [
  {
    name: "get_order_status",
    description: "Customer-এর order এর current status বের করো",
    input_schema: {
      type: "object",
      properties: {
        order_id: {
          type: "string",
          description: "Order ID"
        }
      },
      required: ["order_id"]
    }
  }
]
```

তিনটা জিনিস দিতে হয়:
- **name** — tool এর নাম
- **description** — LLM এটা পড়ে বোঝে কখন call করবে
- **input_schema** — কী parameter লাগবে

Description টা সবচেয়ে important। LLM description পড়েই decide করে কোন tool কখন use করবে।

---

## Tool Execution Flow

```
User: "আমার order 1234 কোথায়?"
         ↓
LLM reads message
         ↓
LLM thinks: "order status দরকার, get_order_status tool আছে"
         ↓
LLM returns: { tool_call: "get_order_status", args: { order_id: "1234" } }
         ↓
তোমার code executes: db.orders.findOne({ id: "1234" })
         ↓
Result: { status: "shipped", courier: "Pathao", eta: "tomorrow" }
         ↓
Result LLM-এ পাঠাও
         ↓
LLM final answer: "আপনার order Pathao-এর মাধ্যমে ship হয়েছে, কাল পৌঁছাবে।"
```

LLM পুরো loop-এ শুধু **decision maker।** Actual কাজ তোমার code করে।

---

## Custom Tools — কী কী বানানো যায়

যা imagine করতে পারো তাই। Rule একটাই — code দিয়ে করা যায়? Tool বানানো যাবে।

**Database Tools:**
```
query_mongodb(collection, filter)
get_slow_queries(threshold_ms)
check_inventory(product_id)
```

**External API Tools:**
```
check_bkash_payment(transaction_id)
get_pathao_tracking(tracking_id)
send_whatsapp_message(phone, message)
```

**File Tools:**
```
read_file(path)
parse_csv(file_path)
extract_pdf_text(file_path)
```

**Infrastructure Tools:**
```
get_ec2_metrics(instance_id)
check_disk_usage(server)
restart_service(service_name)
```

বাইরের কোনো API? সেটাও tool হতে পারে। bKash, Pathao, Steadfast — যেকোনো external service।

---

## External API as Tool — Important Concept

অনেকে মনে করে tool মানে শুধু নিজের database।

না। যেকোনো external API tool হতে পারে।

```typescript
// bKash payment status check করো
{
  name: "check_bkash_payment",
  description: "bKash transaction-এর status verify করো",
  input_schema: {
    properties: {
      transaction_id: { type: "string" }
    }
  }
}

// Implementation:
async function check_bkash_payment({ transaction_id }) {
  const response = await fetch(`https://api.bkash.com/payment/${transaction_id}`, {
    headers: { Authorization: `Bearer ${BKASH_TOKEN}` }
  })
  return response.json()
}
```

LLM bKash API-র কথা জানে না। কিন্তু তুমি tool বানালে — সে জানতে পারবে। এটাই power।

---

## Connectors vs Tools — একটু আলাদা

তুমি হয়তো "Connector" শব্দটা শুনেছো।

```
Tool       = একটা function। শুধু definition।
Connector  = tool + authentication + connection management। পুরো package।
```

Gmail Connector মানে:
```
OAuth handle করে
Token refresh করে
50+ tools expose করে:
  - read_email()
  - send_email()
  - search_emails()
  - create_draft()
```

তুমি শুধু "Connect Gmail" click করো — ভেতরে কী হচ্ছে জানতে হয় না।

**Connector = tools-এর packaged, authenticated, ready-to-use version।**

---

## MCP — Tool-এর Universal Standard

এখানে একটা problem ছিল।

```
তুমি Claude-এর জন্য tool বানালে
→ OpenAI-তে কাজ করে না
→ Gemini-তে কাজ করে না
→ আবার নতুন করে লিখতে হয়
```

প্রতিটা LLM-এর আলাদা format। Tool **LLM-specific** হয়ে যাচ্ছে।

**MCP = Model Context Protocol** এই সমস্যা solve করলো।

Anthropic বানিয়েছে, কিন্তু open standard। USB-C এর মতো ভাবো।

```
আগে:                    MCP দিয়ে:
Tool → Claude only       Tool → যেকোনো LLM
Tool → GPT only    →     Tool → যেকোনো LLM
Tool → Gemini only       Tool → যেকোনো LLM
```

তুমি **একবার MCP Server বানাও** — সব LLM connect করতে পারবে।

```
┌─────────────┐   MCP Protocol   ┌──────────────────────┐
│   LLM Host  │ ◄──────────────► │    MCP Server         │
│             │                  │                       │
│  Claude     │                  │  - query_postgres()   │
│  GPT        │                  │  - send_whatsapp()    │
│  Gemini     │                  │  - check_bkash()      │
└─────────────┘                  └──────────────────────┘
```

MCP-তে তিনটা জিনিস থাকে:

**Tools** — LLM call করতে পারে এমন functions
**Resources** — LLM read করতে পারে এমন data (files, DB)
**Prompts** — reusable prompt templates

---

## এক লাইনে Module 2

```
Tool = function যেটা LLM call করার decision নেয়
     = তোমার code actually execute করে
     = DB হোক, external API হোক — সব same
     
MCP  = সব LLM-এ একই tool চলানোর standard
```

কিন্তু এখন পর্যন্ত একটাই tool call। একটাই request। One shot।

Real কাজ হয় যখন multiple tool calls দরকার, একটার result আরেকটায় লাগে, loop চলে।

এটাই পরের module — **Agent।**
