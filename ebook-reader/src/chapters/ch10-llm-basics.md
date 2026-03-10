# Module 1: LLM — Brain টা আসলে কী?

> "আগে brain টা বোঝো। তারপর সব বাকি জিনিস automatically click করবে।"

---

## LLM কী — একদম সহজ ভাষায়

LLM মানে Large Language Model।

কিন্তু এই definition দিয়ে কিছু বোঝা যায় না। তাই চলো practical ভাবে ভাবি।

তুমি যখন Claude বা ChatGPT-তে কিছু লিখো — সে কিন্তু কোনো database থেকে answer খুঁজে আনছে না। সে কোনো search engine-ও না। সে আসলে একটাই কাজ করছে —

**"পরের word টা কী হওয়া উচিত সেটা predict করছে।"**

ব্যস। এটুকুই।

তুমি লিখলে "আকাশের রং" — সে predict করলো "নীল।"
তুমি লিখলে "Python এ for loop লেখো" — সে predict করলো পুরো code টা।

অবাক লাগছে? একটা simple prediction machine এত কিছু করতে পারে?

হ্যাঁ পারে। কারণ এটা internet-এর প্রায় সব text পড়ে trained। Books, code, papers, conversations — সব। এত data দেখার পর prediction টা এত ভালো হয়ে গেছে যে মনে হয় সে "বুঝছে।"

---

## Text In, Text Out — এটা মাথায় গেঁথে নাও

LLM-এর কাজ একটাই —

```
Input:  Text
Output: Text
```

তুমি text দাও, সে text দেয়। এর বাইরে সে কিছু করে না।

Database query করতে পারে না।
API call করতে পারে না।
File read করতে পারে না।
Internet search করতে পারে না।

এগুলো পরে আসবে — tools দিয়ে। কিন্তু LLM নিজে? শুধু text।

এটা মাথায় না থাকলে পরে অনেক confusion হবে।

---

## Stateless — এটা সবচেয়ে Important Property

তুমি ChatGPT-তে একটা conversation করলে। মনে হলো সে সব মনে রাখছে।

কিন্তু আসলে সে মনে রাখছে না।

প্রতিটা request-এ তুমি যা পাঠাচ্ছো সেটাই সে দেখছে। আগের কথা ভুলে গেছে। পরের request-এ আবার fresh start।

```
Request 1: "আমার নাম Hira"
Response 1: "ঠিক আছে Hira!"

Request 2: "আমার নাম কী?"
Response 2: ???
```

যদি Request 2-এ তুমি আগের conversation পাঠাও — তাহলে সে জানবে।
না পাঠালে? সে জানে না।

এই stateless nature টা পরে context management বোঝার জন্য critical। মাথায় রেখো।

---

## Context Window — তোমার RAM

LLM-এর একটা limit আছে। একবারে কতটুকু text দেখতে পারবে সেটা fixed।

এটাকে বলে **context window।**

```
GPT-4:     128,000 tokens
Claude:    200,000 tokens
Gemini:    1,000,000 tokens
```

Token মানে roughly একটা word বা word-এর অংশ।

মানে হলো — তুমি একবারে এতটুকু text দিতে পারবে। এর বেশি দিলে পুরনোটা বাদ পড়ে যাবে।

এটাকে RAM-এর সাথে তুলনা করো।

RAM যেমন limited — context window তেমন limited।
RAM overflow হলে যেমন পুরনো data যায় — context overflow হলে পুরনো conversation যায়।

এই limit টা agent বানানোর সময় তোমাকে অনেক চিন্তা করাবে। এখন শুধু জেনে রাখো।

---

## LLM কী জানে, কী জানে না

এখানে একটা বড় misconception আছে।

**LLM যা জানে:**
- Training-এর সময় যা দেখেছে তাই
- General knowledge, coding patterns, language, reasoning
- Training cutoff পর্যন্তের information

**LLM যা জানে না:**
- আজকের news কী
- তোমার database-এ কী আছে
- তোমার user-এর order status কী
- Real-time কোনো data

মানে হলো LLM-এর knowledge **frozen।** Training শেষ হওয়ার পর সে আর নতুন কিছু জানে না।

তাহলে real-time data কীভাবে দেবে? Tools দিয়ে। পরের module-এ।

---

## Tokens আর Cost — Developer হিসেবে জানতে হবে

তুমি LLM API use করলে টাকা লাগে। কতটুকু লাগে? Token হিসেবে।

```
Input tokens:  তুমি যা পাঠালে
Output tokens: সে যা reply করলো

দুটোতেই charge হয়।
```

Rough idea:
```
1000 tokens ≈ 750 words ≈ 3-4 paragraphs
```

Claude Sonnet 4.6:
```
Input:  $3 per million tokens
Output: $15 per million tokens
```

Agent বানালে tool calls-এ অনেক tokens যায়। Cost optimize করা একটা real engineering problem।

---

## এক লাইনে Module 1

```
LLM = brilliant brain
    + frozen knowledge (training cutoff পর্যন্ত)
    + stateless (প্রতিটা call fresh)
    + limited context (RAM এর মতো)
    + text in, text out only
```

এই limitations গুলো weakness না — এগুলো বোঝার পরেই তুমি বুঝবে tools, agents, RAG কেন দরকার।

পরের module-এ দেখবো — এই blind, stateless brain-কে কীভাবে হাত-পা দেওয়া যায়।
