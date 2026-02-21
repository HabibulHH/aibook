# Chapter 1: Introduction — Tools দিয়ে শুরু

> "মডেল কী? ওটা পরে বলবো। আগে বলো এই জিনিসগুলো দিয়ে তুমি কী করতে পারো।"

---

## তো শুরু করা যাক

তুমি যদি এই বইটা হাতে নিয়েছো, তাহলে তুমি হয়তো একজন ডেভেলপার, বা ডেভেলপার হতে চাও। বা হয়তো এমন কেউ যে AI জিনিসটা বুঝতে চাও, কিন্তু কোথা থেকে শুরু করবো বুঝতে পারছো না। এই চ্যাপ্টারটা তোমার জন্য।

আমি এখানে তোমাকে "Large Language Model কী?" বা "Transformer Architecture কিভাবে কাজ করে?" এইসব দিয়ে শুরু করবো না। কারণ সেটা করলে তুমি প্রথম পেজেই বোর্ড হয়ে যাবে। বরং, আমরা একটা অন্য পথে যাবো।

আমরা শুরু করবো tools দিয়ে। মানে, তুমি প্রতিদিন যেই জিনিসগুলো use করো বা করতে পারো — সেগুলো দিয়ে শুরু। তারপর ধীরে ধীরে বুঝবো এসবের পেছনে কী কাজ করছে। Deal?

---

## তুমি প্রতিদিন যেই AI Tools ব্যবহার করো (বা করতে পারো)

একটু ভাবো। তুমি সকালে উঠে কোড লিখতে বসলে। VS Code open করলে। Cursor use করলে হয়তো। বা GitHub Copilot on আছে। তুমি একটা function লিখতে শুরু করলে, সেটা নিজে থেকেই suggestion দিচ্ছে। এটা AI।

এরপর তুমি একটা bug পেলে, ChatGPT তে গিয়ে paste করলে। বা Claude এ গিয়ে জিজ্ঞেস করলে। সেটাও AI। বিকেলে তুমি Perplexity তে search করলে "NestJS rate limiting best practices"। সেটাও AI।

মানে AI জিনিসটা এখন আর শুধু একটা research paper বা textbook এর topic না। এটা তোমার daily workflow এর পার্ট। তো চলো দেখি কোন কোন tools আছে যেগুলো তোমার কাজে লাগবে।

---

## ১. Chat Tools — যেখানে তুমি কথা বলে কাজ করাও

এইগুলো হলো সবচেয়ে common AI tools যেগুলো মানুষ প্রতিদিন ব্যবহার করে। তুমি একটা prompt লিখো, উত্তর পাও। একটু দেখো:

**ChatGPT (OpenAI)** — সবচেয়ে popular। Code লিখতে পারে, explain করতে পারে, email draft করতে পারে, এমনকি image ও generate করতে পারে। এর পেছনের model হলো **GPT-4o** এবং **o3/o4-mini** (reasoning model)।

**Claude (Anthropic)** — লম্বা code বোঝার জন্য অনেকের পছন্দ। বড় ফাইল দিতে পারো, document analyze করতে পারো। এর পেছনের model হলো **Claude Opus, Sonnet, Haiku**।

**Gemini (Google)** — Google এর AI। Google Search এর সাথে integrated। এর পেছনে আছে **Gemini 2.5 Pro** এবং **Flash** model।

**Perplexity** — AI-powered search engine। Google এর বদলে এটা সরাসরি উত্তর দেয় source সহ। এটা GPT, Claude, এবং নিজেদের **Sonar** model mix করে ব্যবহার করে।

---

## ২. Code Assistant Tools — তোমার Editor এর ভেতরের AI

এইগুলো একটু আলাদা। এগুলো তোমার code editor এর ভেতরে বসে কাজ করে। তুমি লিখছো, সেটা বুঝে ফেলছে তুমি কী লিখতে চাও, এবং suggestion দিচ্ছে।

**GitHub Copilot** — VS Code এর ভেতরে autocomplete এর মতো কাজ করে। এখন এটা একটা model এ সীমাবদ্ধ না — **GPT-4o, Claude, Gemini** সব ব্যবহার করে।

**Cursor** — VS Code এর মতো দেখতে কিন্তু ভেতরে AI built-in। তুমি নিজে বেছে নিতে পারো কোন model use করবে — Claude, GPT, Gemini। এইটাই interesting পার্ট — tool একটা, কিন্তু পেছনের engine বদলানো যায়।

**v0 by Vercel** — prompt দিলে UI component generate করে দেয়। "একটা login page বানাও dark theme এ" — এইটুকু বললেই হয়ে যায়। Mainly **Claude** powered।

**Bolt / Lovable** — prompt থেকে full app scaffold করে দেয়। Claude + GPT mixed ব্যবহার করে।

---

## ৩. Image Generation Tools — লিখে বলো, ছবি পাও

এই category টা মজার। তুমি একটা prompt লিখবে, যেমন "a futuristic city at sunset, cyberpunk style" — এবং সেই মতো একটা image generate হয়ে যাবে।

**Midjourney** — সবচেয়ে artistic quality। নিজস্ব proprietary model।

**DALL-E 3** — OpenAI এর image model। ChatGPT এর ভেতরেই ব্যবহার করতে পারো।

**Stable Diffusion** — open source! নিজের computer এ চালাতে পারো। Stability AI এর model।

**Flux** — Black Forest Labs এর model। নতুন, কিন্তু quality ভালো।

---

## ৪. Voice এবং Video Tools

এই জায়গাটা অনেকের কাছে নতুন। কিন্তু এখানেই AI এর সবচেয়ে "wow" moment গুলো ঘটছে।

**ElevenLabs** — text দিলে human-like voice এ বলে দেয়। Voice clone ও করতে পারে। ভাবো, তোমার blog post টা অটোমেটিক podcast এ কনভার্ট হয়ে গেলো।

**Whisper (OpenAI)** — উল্টোটা। voice কে text এ কনভার্ট করে (speech-to-text)। বাংলায়ও ভালো কাজ করে।

**Runway Gen-3** — text/image থেকে video generate করে। শর্ট ক্লিপ, ad, concept video এর জন্য ব্যবহার হয়।

**Suno** — prompt দিলে গান বানিয়ে দেয়। Lyrics সহ। মজার জিনিস কিন্তু।

---

## ৫. Open Source Models — যেগুলো সবার জন্য খোলা

এইটা একটা গুরুত্বপূর্ণ পয়েন্ট। উপরের tools গুলোর বেশিরভাগই proprietary — মানে তুমি ওদের platform এ গিয়ে use করতে হবে। কিন্তু কিছু model আছে যেগুলো open source — মানে তুমি চাইলে নিজের server এ চালাতে পারো।

**Meta Llama 4** — Meta এর open source model। অনেক startup এবং tool এর পেছনে Llama চলছে।

**Mistral** — France এর company। Mistral Large এবং **Codestral** (code-focused model) দিয়ে পরিচিত।

এই open source model গুলো তুমি Ollama, Hugging Face, বা Together AI এর মতো platform দিয়ে run করতে পারো। পরে আমরা এটা নিয়ে আরও কথা বলবো।

---

## বড় কথাটা কী?

এখানে একটা গুরুত্বপূর্ণ জিনিস বুঝে নাও। বেশিরভাগ AI tools এর পেছনে একই কয়েকটা model ঘুরছে। Cursor এর পেছনে Claude আছে। ChatGPT এর পেছনে GPT-4o আছে। v0 এর পেছনেও Claude। GitHub Copilot এখন একাধিক model support করে।

মানে tool আলাদা, model আলাদা।

Tool হলো গাড়ি। Model হলো engine। তুমি Toyota চালাও বা BMW, দুটোই গাড়ি। কিন্তু engine আলাদা, performance আলাদা। AI এর ক্ষেত্রেও ঠিক তাই।

---

## একনজরে দেখে নাও: Tools এবং তাদের পেছনের Models

| Tool | Category | Model / Engine |
|------|----------|---------------|
| ChatGPT | Chat | GPT-4o, o3, o4-mini |
| Claude | Chat | Opus, Sonnet, Haiku |
| Gemini | Chat | Gemini 2.5 Pro, Flash |
| Perplexity | Search | GPT + Claude + Sonar |
| GitHub Copilot | Code | GPT-4o, Claude, Gemini |
| Cursor | Code | Claude, GPT, Gemini |
| v0 (Vercel) | Code/UI | Claude |
| Bolt / Lovable | Code/App | Claude + GPT |
| Midjourney | Image | Proprietary |
| DALL-E 3 | Image | DALL-E 3 (OpenAI) |
| Stable Diffusion | Image | Open Source Diffusion |
| Flux | Image | Black Forest Labs |
| ElevenLabs | Voice | Proprietary TTS |
| Whisper | Speech-to-Text | Whisper v3 (OpenAI) |
| Runway | Video | Gen-3 Alpha |
| Suno | Music | Proprietary |
| Llama 4 | Open Source LLM | Meta |
| Mistral / Codestral | Open Source LLM | Mistral AI |

---


## কিন্তু এই Model জিনিসটা আসলে কী? ভেতরে কী আছে?

তুমি এতক্ষণ দেখলে কোন tool এর পেছনে কোন model আছে। কিন্তু একটা প্রশ্ন মাথায় আসতে পারে — "model বলতে আসলে কী বোঝায়? ভেতরে কী থাকে? Code? Math? Logic?"

উত্তরটা অনেকের কাছে anticlimactic লাগতে পারে।

একটা model file open করলে তুমি পাবে শুধু **numbers**। Billions of numbers। কোনো code নাই, কোনো if-else নাই, কোনো rule নাই। শুধু সংখ্যা।

এই numbers গুলোকে বলে **weights** বা **parameters**। এগুলো সাজানো থাকে **matrices** (2D number grid) আর **tensors** (multi-dimensional arrays) হিসেবে। যেমন একটা 7B model মানে — প্রায় ৭০০ কোটি (7 billion) floating-point numbers, layer এর পর layer সাজানো।

```
Model file এর ভেতর:

Layer 1:
  - attention.query_weight:  [4096 x 4096] matrix → ১.৬ কোটি numbers
  - attention.key_weight:    [4096 x 4096] matrix → ১.৬ কোটি numbers
  - attention.value_weight:  [4096 x 4096] matrix → ১.৬ কোটি numbers
  - feedforward.weight:      [4096 x 11008] matrix → ৪.৫ কোটি numbers

Layer 2: (same structure)
Layer 3: (same structure)
... 32 layers ...

Total: ~৭০০ কোটি numbers = 7B parameter model
```

### তাহলে কাজ করে কিভাবে?

Model যখন run করে, সে একটাই কাজ করে বারবার — **matrix multiplication**। তোমার input text কে numbers এ convert করে (tokenization + embedding), তারপর সেই numbers কে layer এর পর layer multiply করে, শেষে output বের করে।

```
তোমার text → numbers → multiply (layer 1) → multiply (layer 2) → ... 32 বার ... → output number → পরের word
```

কোনো magic নাই। শুধু যোগ-গুণ। কিন্তু ৭০০ কোটি number দিয়ে ৩২টা layer এ যোগ-গুণ করলে — intelligent behavior emerge করে। এইটাই fascinating part।

### "Intelligence" কোথায় তাহলে?

Intelligence আছে weight values এ — ঐ ৭০০ কোটি numbers এ। Training এর সময় model কোটি কোটি text পড়ে এই numbers একটু একটু করে adjust করেছে। "The capital of Bangladesh is" এর পরে "Dhaka" আসা উচিত — এটা কোনো rule লিখে দেওয়া হয়নি, weights নিজে শিখেছে।

একটা analogy দিই। Piano তে ৮৮টা key আছে — সেটা হলো architecture। কিন্তু কোন key কখন কতটা জোরে চাপতে হবে — সেটা musician এর হাতে শেখা। Model এর weights হলো সেই "শেখা"।

তাহলে সোজা কথায়:

> **Model = Architecture (কোন order এ multiply করবে) + Weights (multiply করার numbers)**

Architecture সবার একই হতে পারে — GPT, LLaMA দুটোই Transformer architecture। কিন্তু weights আলাদা, কারণ আলাদা data দিয়ে train হয়েছে। তাই performance আলাদা।

### কিন্তু দাঁড়াও — training এর আগে কী থাকে?

এখন তোমার মাথায় আসতে পারে — "training data না থাকলে তো model এর ভেতরে কিছুই নাই। তাহলে raw model টা কী জিনিস?"

উত্তর: raw model এও matrices আছে, layers আছে, structure আছে — হুবহু same shape। কিন্তু numbers গুলো হলো **সম্পূর্ণ random**। কোনো intelligence নাই। কোনো meaning নাই।

```python
import torch

# এইটা হলো "raw" untrained weight — purely random
random_weight = torch.randn(4096, 4096)
# tensor([ 0.2104, -1.3452,  0.8821, -0.0034,  1.2290, ...])
# meaningless numbers
```

এই অবস্থায় model কে "What is Bangladesh?" জিজ্ঞেস করলে সে বলবে "purple fish dancing moon table" — random token এর পর random token। কারণ ভেতরের numbers এর কোনো pattern নাই।

### Training এ কী হয়?

Model কোটি কোটি text পড়ে, আর প্রতিবার ঐ random numbers একটু একটু **adjust** হয়:

```
Step 1: "The capital of Bangladesh is ___"
  → model guess করলো "potato" (random weight তো!)
  → WRONG! ভুলের পরিমাণ (loss) অনেক বেশি
  → সব weights একটু adjust হলো

Step 1000: same ধরনের text আবার দেখলো
  → model guess করলো "Delhi"
  → কাছাকাছি, কিন্তু still wrong
  → weights আবার একটু adjust

Step 1,000,000:
  → model guess করলো "Dhaka"
  → CORRECT! Loss near zero
  → weights stable
```

এই process কে বলে **gradient descent** — ভুল কমানোর জন্য weights এর value ক্ষুদ্র ক্ষুদ্র পরিমাণে change করা। একটা step এ tiny change — কিন্তু billions of steps পরে random numbers meaningful হয়ে যায়।

### Training Data কি Model এর ভেতরে থাকে?

না। Model file open করলে কোনো text পাবে না, কোনো document পাবে না, কোনো training data নাই।

কিন্তু training data **indirectly encoded** থাকে weight values এ।

ভাবো — তুমি পরীক্ষার আগে ১০টা বই পড়লে। পরীক্ষার হলে তুমি বই নিয়ে যাও না। কিন্তু বই পড়ে যা শিখেছো — সেটা তোমার brain এ আছে। Model এর weights হলো সেই brain।

তাই GPT আর LLaMA দুটোই same Transformer architecture, same shape এর matrices — কিন্তু আলাদা data দিয়ে train হয়েছে। তাই weight values আলাদা, তাই একজন code এ ভালো, আরেকজন reasoning এ ভালো।

### একনজরে

| State | Matrices আছে? | Numbers কেমন? | Intelligence আছে? |
|-------|---------------|---------------|-------------------|
| Raw / Untrained | ✅ same shape | Random garbage | ❌ শূন্য |
| After Training | ✅ same shape | Meaningful, tuned | ✅ Data থেকে শেখা |

> **Raw model = architecture + random floating points। Trained model = same architecture + meaningful floating points। Structure same, শুধু numbers এর value বদলেছে — আর সেটাই সব।**

## পরের Chapter এ কী আছে?

পরের chapter এ আমরা দেখবো — এই model গুলো raw data নিয়ে সরাসরি কাজ করে না। Data model এ ঢোকার আগে একটা পুরো pipeline আছে। সেই pipeline টা কী, data কিভাবে clean হয়, transform হয়, আর model পর্যন্ত পৌঁছায় — সেটা দেখবো।

আজকের জন্য এটুকুই মনে রাখো: তুমি tools চেনো, tools use করো, এবং জেনে রাখো কোন tool এর পেছনে কোন model কাজ করছে। এটাই তোমার AI journey এর শুরু।
