# Chapter 4: Model এর ভেতরে — LLM, VLM, Diffusion, Token

> "Tool চিনলে, model চিনলে। এখন বুঝো এই model গুলো আসলে কিভাবে কাজ করে।"

---

## আগের Chapter থেকে যেখানে ছিলাম

আগের তিনটা chapter এ তুমি দুইটা জিনিস শিখেছো। প্রথমত, কোন কোন AI tools আছে এবং সেগুলোর পেছনে কোন model কাজ করছে। দ্বিতীয়ত, model কিভাবে prepare করে এবং Hugging Face এ store করে।

এখন প্রশ্ন হলো — এই model গুলো আসলে কী জিনিস? ভেতরে কী হচ্ছে? ChatGPT কে তুমি যখন কিছু জিজ্ঞেস করো, সে কিভাবে উত্তর দেয়? Midjourney কিভাবে ছবি বানায়? Whisper কিভাবে তোমার কথা বুঝতে পারে?

চলো ভাঙি।

---

## Token — সবকিছুর শুরু এখান থেকে

AI model কে কিছু বোঝানোর আগে একটা fundamental জিনিস বুঝতে হবে — **Token**।

তুমি আর আমি text পড়ি word ধরে ধরে। কিন্তু AI model text পড়ে token ধরে ধরে। Token হলো text এর ছোট ছোট টুকরো। কখনো একটা পুরো word একটা token, কখনো একটা word এর অংশ একটা token।

উদাহরণ দিই:

```
"I love programming" → ["I", " love", " program", "ming"]
```

দেখো, "programming" word টা দুইটা token এ ভাগ হয়ে গেছে — "program" আর "ming"। এটা model নিজে ঠিক করে কিভাবে ভাগ করবে। এইটাকে বলে **Tokenization**।

বাংলায়?

```
"আমি ভাত খাই" → ["আমি", " ভ", "াত", " খ", "াই"]
```

দেখো, বাংলায় আরও বেশি token লাগছে। কারণ বেশিরভাগ model ইংরেজি data দিয়ে বেশি train হয়েছে, তাই ইংরেজি word গুলো efficiently tokenize হয়। বাংলা বা অন্য ভাষায় বেশি token লাগে। এটা একটা real problem — বাংলায় same কথা বলতে বেশি token খরচ হয়, মানে বেশি টাকা লাগে API call এ।

### Token কেন important?

কারণ AI এর পুরো দুনিয়া token ঘিরে চলে:

**Context Window** — একটা model একবারে কতো token handle করতে পারে। GPT-4o এর context window 128K tokens। Claude এর 200K tokens। মানে তুমি এতো বড় text একবারে দিতে পারো। এর বেশি দিলে? সে ভুলে যাবে আগের কথা।

**Pricing** — API ব্যবহার করলে টাকা লাগে token হিসেবে। Input tokens আর output tokens আলাদা আলাদা price। তাই prompt ছোট রাখলে টাকা বাঁচে।

**Speed** — বেশি token মানে বেশি সময়। তাই model এর response যতো লম্বা, ততো slowly আসে।

একটা rough calculation: ইংরেজিতে ১ token ≈ ৪ characters ≈ ০.৭৫ word। মানে ১০০০ token ≈ ৭৫০ word।

---

## LLM — Large Language Model

এখন আসো মূল জিনিসে। **LLM** মানে Large Language Model। এইটা হলো সেই ধরনের model যেটা text বোঝে এবং text generate করে। ChatGPT, Claude, Gemini, Llama — এরা সবাই LLM।

### LLM আসলে কী করে?

খুব সোজা করে বললে — LLM একটাই কাজ করে: **পরের token predict করা**।

তুমি যখন লিখো "The capital of Bangladesh is", model ভাবে — এর পরের token কী হওয়া উচিত? সে তার training data থেকে শিখেছে যে এই pattern এর পরে "Dhaka" আসার probability সবচেয়ে বেশি। তাই সে "Dhaka" output দেয়।

ব্যস। এইটুকুই। পুরো ChatGPT, পুরো Claude — সবকিছু এই একটা principle এর উপর দাঁড়িয়ে আছে: **next token prediction**।

"কিন্তু ভাই, শুধু পরের word guess করে এতো intelligent হয় কিভাবে?"

ভালো প্রশ্ন। কারণ এই "guess" করাটা simple না। Model টা billions of parameters এর মধ্য দিয়ে calculate করে বের করে কোন token আসা উচিত। এই parameters গুলো training এর সময় শিখেছে internet এর কোটি কোটি text পড়ে। তাই সে শুধু word match করে না — সে context বোঝে, logic বোঝে, pattern বোঝে।

### Transformer — LLM এর ভেতরের Engine

LLM এর ভেতরে যেই architecture কাজ করে সেটার নাম **Transformer**। ২০১৭ সালে Google এর একটা paper এ এটা introduce হয়েছিলো — "Attention Is All You Need"। এই paper টা AI এর ইতিহাস বদলে দিয়েছে।

Transformer এর key idea হলো **Attention**। মানে model যখন একটা word process করছে, সে শুধু আগের word দেখে না — পুরো context এর সব word এর দিকে "attention" দেয়। কোন word কতোটা important সেটা calculate করে।

ধরো sentence হলো: "The cat sat on the mat because it was tired."

"it" বলতে কী বোঝাচ্ছে? Cat না mat? Model Attention mechanism ব্যবহার করে বুঝবে "it" আসলে "cat" কে refer করছে। কারণ "tired" হওয়ার context এ "cat" বেশি relevant।

এইটা তুমি deep dive করতে চাইলে পরে করতে পারো। আপাতত এটুকু জানলেই চলবে: **Transformer architecture + Attention mechanism = LLM এর brain**।

### LLM এর Types

সব LLM একরকম না। কয়েকটা ভাগ আছে:

**Base Model** — শুধু text prediction শিখেছে। তুমি যদি এটাকে প্রশ্ন করো, সে হয়তো প্রশ্নের উত্তর না দিয়ে প্রশ্নের পরে আরেকটা প্রশ্ন generate করবে। কারণ সে internet এ দেখেছে প্রশ্নের পরে আরেকটা প্রশ্ন আসে।

**Instruction-tuned Model** — base model কে instruction follow করতে শেখানো হয়েছে। "Summarize this" বললে summarize করবে, "Translate this" বললে translate করবে। ChatGPT, Claude — এরা সবাই instruction-tuned।

**Chat Model** — instruction-tuned এর উপর আরেকটা layer। Multi-turn conversation handle করতে পারে। তুমি ১০টা message পাঠালে সে সব মনে রাখে (context window এর মধ্যে)।

**Reasoning Model** — নতুন category। o3, o4-mini এরা এই ধরনের। এরা উত্তর দেওয়ার আগে "think" করে — step by step reason করে, তারপর answer দেয়। Math, logic, complex problem এ এরা বেশি ভালো।

---

## VLM — Vision Language Model

LLM শুধু text বোঝে। কিন্তু তুমি যদি ChatGPT তে একটা screenshot দাও আর বলো "এই error টা কী?" — সে বুঝতে পারে। কিভাবে?

কারণ এটা শুধু LLM না, এটা **VLM — Vision Language Model**। মানে text আর image দুটোই বোঝে।

VLM কিভাবে কাজ করে? Simple ভাবে বললে — image কে একটা **Vision Encoder** দিয়ে process করা হয়। এই encoder image এর features বের করে token এর মতো format এ convert করে। তারপর সেই visual tokens আর text tokens একসাথে LLM এ ঢুকে। LLM তখন দুটো মিলিয়ে বোঝে।

```
Image → Vision Encoder → Visual Tokens
                                          ↘
                                           LLM → Response
                                          ↗
Text Prompt → Tokenizer → Text Tokens
```

### VLM Examples

**GPT-4o** — text + image + audio সব বোঝে। এইটাকে বলে multimodal।

**Claude (Vision)** — image analyze করতে পারে। Screenshot দিলে UI বুঝতে পারে, document পড়তে পারে।

**Gemini** — Google এর model, natively multimodal। Image, video, audio সব handle করে।

**LLaVA** — open source VLM। Llama base এ vision capability add করা হয়েছে।

### কোথায় কাজে লাগে?

তুমি একজন developer হিসেবে VLM কোথায় ব্যবহার করতে পারো?

**UI screenshot থেকে code** — একটা Figma design এর screenshot দাও, VLM সেটা দেখে HTML/React code লিখে দেবে।

**Document parsing** — invoice, receipt, form এর photo থেকে data extract করো।

**Error debugging** — error screenshot দিলে কী problem বুঝিয়ে দেয়।

**Chart/Graph analysis** — একটা chart এর image দিলে data explain করে দেয়।

---

## Diffusion Model — যেভাবে ছবি তৈরি হয়

Midjourney, DALL-E, Stable Diffusion — এরা সবাই **Diffusion Model** ব্যবহার করে। এইটা LLM থেকে সম্পূর্ণ আলাদা কাজ করে।

### Idea টা কী?

একটা মজার analogy দিই। ধরো তোমার কাছে একটা পরিষ্কার ছবি আছে। এখন তুমি সেই ছবিতে ধীরে ধীরে noise (ঝামেলা) add করলে। আরও noise, আরও noise — একসময় পুরো ছবি complete noise হয়ে গেলো। TV এর static এর মতো।

Diffusion Model এর training হলো এই process টা উল্টো করা শেখা। মানে noise থেকে ধীরে ধীরে noise সরিয়ে পরিষ্কার ছবি বানানো।

```
Training:
পরিষ্কার ছবি → noise add → noise add → noise add → pure noise
                    ↕ (model শেখে এই process উল্টো করতে)
Generation:
pure noise → denoise → denoise → denoise → নতুন ছবি!
```

### Text থেকে Image কিভাবে?

তুমি যখন লিখো "a cat sitting on the moon" — তখন কী হয়?

**Step 1:** তোমার text prompt টা একটা **Text Encoder** (সাধারণত CLIP model) দিয়ে process হয়। এটা text কে একটা numerical representation এ convert করে।

**Step 2:** এই representation দিয়ে Diffusion Model কে guide করা হয়। Model random noise থেকে শুরু করে, step by step denoise করে, কিন্তু প্রতি step এ text guidance follow করে — "cat থাকতে হবে", "moon থাকতে হবে"।

**Step 3:** অনেকগুলো step পরে — একটা ছবি বেরিয়ে আসে যেটা তোমার prompt এর সাথে match করে।

### Key Terms

**Steps** — কতো বার denoise করবে। বেশি steps = better quality কিন্তু slow। সাধারণত ২০-৫০ steps যথেষ্ট।

**CFG Scale (Classifier-Free Guidance)** — text prompt কতোটা strictly follow করবে। High CFG = prompt এর খুব কাছাকাছি থাকবে। Low CFG = model নিজের creative freedom নেবে।

**Seed** — random starting noise এর number। Same seed + same prompt = same image। তাই তুমি একটা ভালো result পেলে seed save করে রাখতে পারো।

**Sampler** — noise remove করার method। Euler, DPM++, DDIM — এরকম অনেক sampler আছে। প্রতিটা একটু আলাদা result দেয়।

---

## Embedding — Text কে Number এ রূপান্তর

এইটা একটু abstract কিন্তু অনেক important concept। AI model text সরাসরি বোঝে না — সে number বোঝে। তাই text কে number এ convert করতে হয়। এইটাকে বলে **Embedding**।

একটা word বা sentence কে একটা long list of numbers (vector) এ convert করা হয়। যেমন:

```
"king"  → [0.2, 0.8, -0.1, 0.5, ...]  (হয়তো 768 টা number)
"queen" → [0.3, 0.7, -0.1, 0.6, ...]
"car"   → [-0.5, 0.1, 0.9, -0.3, ...]
```

দেখো, "king" আর "queen" এর numbers কাছাকাছি। কারণ মানে কাছাকাছি। "car" এর numbers অনেক আলাদা। কারণ meaning আলাদা।

### Embedding কোথায় কাজে লাগে?

**Semantic Search** — তুমি একটা search system বানাতে চাও যেটা শুধু keyword match না, meaning ও বোঝে। যেমন "cheap flights" search করলে "affordable air tickets" ও আসবে। Embedding দিয়ে এটা possible।

**RAG (Retrieval Augmented Generation)** — তোমার নিজের documents আছে, তুমি চাও AI সেগুলো পড়ে উত্তর দিক। প্রথমে documents কে embedding করে vector database এ রাখো (Pinecone, Weaviate, ChromaDB)। তারপর user এর question এর embedding match করো documents এর সাথে। Related documents পেলে সেগুলো LLM কে দাও context হিসেবে।

**Recommendation** — user এর interest আর content এর embedding compare করে recommend করা যায়।

এই concept টা developer হিসেবে তোমার অনেক কাজে আসবে। RAG system বানানো এখন অনেক common — এবং এর core হলো embedding।

---

## Temperature, Top-p — Model এর Creativity Control

ChatGPT বা Claude ব্যবহার করলে তুমি হয়তো দেখেছো — কখনো model অনেক creative answer দেয়, কখনো অনেক boring আর predictable। এটা control করে কিছু parameters:

### Temperature

এইটা সবচেয়ে important parameter।

**Temperature = 0** → model সবসময় সবচেয়ে probable token pick করবে। মানে একই prompt দিলে একই answer আসবে। Deterministic। Factual questions এর জন্য ভালো।

**Temperature = 1** → normal randomness। Balanced creativity।

**Temperature = 1.5+** → অনেক random, creative, কিন্তু ভুলভাল কথাও বলতে পারে।

ভাবো এভাবে — Temperature হলো model এর "মাতাল meter"। কম temperature = সিরিয়াস, focused। বেশি temperature = loose, creative, কিন্তু মাঝে মাঝে আজেবাজে।

### Top-p (Nucleus Sampling)

এইটা আরেকটা way to control randomness। Top-p = 0.9 মানে model শুধু সেই tokens consider করবে যেগুলোর cumulative probability ৯০% cover করে। বাকি rare tokens ignore করবে।

### বাস্তবে কিভাবে ব্যবহার করবে?

```python
# API call এ এভাবে set করো
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a poem about coding"}],
    temperature=0.7,    # moderate creativity
    top_p=0.9,          # ignore very rare tokens
    max_tokens=500,     # সর্বোচ্চ কতো token output দেবে
)
```

| Use Case | Temperature | Top-p |
|----------|-------------|-------|
| Code generation | 0 - 0.2 | 0.9 |
| Factual Q&A | 0 - 0.3 | 0.9 |
| Creative writing | 0.7 - 1.0 | 0.95 |
| Brainstorming | 1.0 - 1.5 | 1.0 |

---

## Prompt Engineering — Model কে ঠিকমতো কাজ করানো

Model আছে, parameters বুঝেছো। এখন সবচেয়ে practical skill — **Prompt Engineering**। মানে model কে কিভাবে instruct করলে সে সবচেয়ে ভালো output দেয়।

### Rule 1: Specific হও

```
❌ "Write something about Bangladesh"
✅ "Write a 200-word paragraph about Bangladesh's tech startup ecosystem in 2025, 
   focusing on fintech and edtech sectors"
```

### Rule 2: Role দাও

```
✅ "You are a senior backend engineer with 10 years of experience in NestJS. 
   Review this code and suggest improvements for performance..."
```

Model কে একটা role দিলে সে সেই perspective থেকে answer দেয়। অনেক বেশি relevant হয়।

### Rule 3: Format specify করো

```
✅ "List the top 5 reasons, each with a one-line explanation. 
   Output as a numbered list."
```

```
✅ "Respond in JSON format with keys: title, summary, tags"
```

### Rule 4: Example দাও (Few-shot)

```
✅ "Convert these Bengali sentences to formal English.

   Bengali: আমি কাল আসবো
   English: I will arrive tomorrow.
   
   Bengali: সে খুব ভালো ছেলে
   English: He is a very good person.
   
   Bengali: বাজারে যেতে হবে
   English:"
```

Model example দেখে pattern বুঝে ফেলে এবং সেই pattern follow করে। এইটাকে বলে **Few-shot Prompting**।

### Rule 5: Chain of Thought

Complex problem এ model কে বলো step by step ভাবতে:

```
✅ "Think step by step. First analyze the problem, then list possible solutions, 
   then pick the best one with reasoning."
```

এটা especially math, logic, আর complex reasoning এ অনেক ভালো কাজ করে।

### Rule 6: System Prompt ব্যবহার করো

API ব্যবহার করলে system prompt দিতে পারো — এটা model এর "personality" set করে:

```python
messages=[
    {"role": "system", "content": "You are a Bengali language tutor. 
     Always respond in Bengali. Keep explanations simple."},
    {"role": "user", "content": "Explain recursion to me"}
]
```

---

## একনজরে: Model Types এবং কী করে

| Model Type | কী বোঝে | কী Output দেয় | Example |
|-----------|---------|--------------|---------|
| LLM | Text | Text | GPT-4o, Claude, Llama |
| VLM | Text + Image | Text | GPT-4o Vision, LLaVA |
| Diffusion | Text (prompt) | Image | Stable Diffusion, DALL-E |
| TTS | Text | Audio/Voice | ElevenLabs |
| STT | Audio | Text | Whisper |
| Music | Text | Music | Suno |
| Video | Text/Image | Video | Runway Gen-3 |

---

## এই Chapter এ কী শিখলে?

অনেক কিছু শিখলে আজকে। Token কী আর কেন important — এটা বুঝলে। LLM কিভাবে next token predict করে — এটা বুঝলে। Transformer আর Attention mechanism এর basic idea পেলে। VLM কিভাবে image আর text একসাথে বোঝে — দেখলে। Diffusion Model কিভাবে noise থেকে ছবি বানায় — জানলে। Embedding কী আর কোথায় কাজে লাগে — শিখলে। Temperature আর Top-p দিয়ে model control করা — বুঝলে। আর Prompt Engineering এর ৬টা practical rule পেলে।

এই chapter এ theory একটু বেশি ছিলো, মানি। কিন্তু এই foundation না থাকলে পরের chapters বুঝবে না। পরের chapter থেকে আমরা পুরো hands-on — PyTorch, TensorFlow, ONNX, llama.cpp — কে কী, কোথায় কাজে লাগে।

তোমার homework? ChatGPT বা Claude এ গিয়ে temperature নিয়ে experiment করো। একই prompt দিয়ে temperature 0 এ আর temperature 1.5 এ output compare করো। পার্থক্যটা নিজে দেখো। এটাই সবচেয়ে ভালো শেখা।
