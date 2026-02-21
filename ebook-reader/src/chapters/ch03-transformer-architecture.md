# Chapter 3: Transformer Architecture — ভেতরে কী আছে?

> "সবাই বলে Transformer, Transformer। কিন্তু এই জিনিসটা আসলে কী? GPT, BERT, LLaMA — এরা কি সব same?"

---

## Transformer এর আগে কী ছিলো?

Transformer বুঝতে হলে আগে জানতে হবে — এর আগে কী ব্যবহার হতো, আর কেন সেগুলো replace হয়ে গেলো।

AI/ML এ model architecture এর একটা evolution আছে। এটা একটু দেখো:

**MLP (Feedforward Neural Network)** — সবচেয়ে simple। Data এক দিকে যায় — input → hidden layers → output। Tabular data তে কাজ করতো।

**CNN (Convolutional Neural Network)** — image এর জন্য আসলো। Local pattern detect করতে পারতো — edge, shape, object। Image classification এ revolution আনলো।

**RNN (Recurrent Neural Network)** — sequential data এর জন্য — text, audio, time series। একটা word পড়ে, মনে রাখে, পরের word পড়ে। কিন্তু সমস্যা ছিলো — লম্বা sequence হলে আগের কথা "ভুলে যেতো" (vanishing gradient problem)।

**LSTM (Long Short-Term Memory)** — RNN এর "ভুলে যাওয়া" সমস্যা fix করার চেষ্টা করলো। একটা memory cell add করলো। Better হলো, কিন্তু still slow ছিলো — কারণ word গুলো একটার পর একটা process করতো, parallel করতে পারতো না।

তারপর ২০১৭ সালে এলো **Transformer**।

---

## Transformer কী?

২০১৭ সালে Google এর researchers একটা paper publish করলো — **"Attention Is All You Need"**। এই paper AI এর ইতিহাস বদলে দিয়েছে।

Transformer এর core idea হলো — word গুলো একটার পর একটা পড়ার দরকার নাই। পুরো sequence একবারে দেখো, আর **Self-Attention** দিয়ে বোঝো কোন word কার সাথে কতটুকু related।

RNN যদি হয় বই line by line পড়া, Transformer হলো পুরো page একবারে দেখে relationship বোঝা।

এর ফলে কী হলো?

**Parallel processing** — সব word একসাথে process হয়, তাই training অনেক fast।

**Long-range understanding** — ১০০০ word আগের context ও মনে থাকে।

**Scalable** — বেশি data, বেশি GPU দিলে আরও ভালো হয়। এই কারণেই billion-parameter models সম্ভব হলো।

---

## Self-Attention — Transformer এর Brain

Transformer এর সবচেয়ে important part হলো **Self-Attention mechanism**। এটা কিভাবে কাজ করে?

ধরো একটা sentence: "The server crashed because it ran out of memory."

"it" বলতে কী বোঝাচ্ছে? "server" নাকি "memory"? তুমি আর আমি বুঝি "it" = "server"। কিন্তু model কিভাবে বুঝবে?

Self-Attention এ প্রতিটা word প্রতিটা word এর দিকে "তাকায়" এবং calculate করে — "আমার সাথে কে কতটুকু related?" "it" যখন process হচ্ছে, সে দেখবে "server" এর সাথে তার attention score সবচেয়ে বেশি। তাই সে বুঝে যাবে "it" মানে "server"।

Technical ভাবে এটা ৩টা জিনিস দিয়ে কাজ করে:

**Query (Q)** — "আমি কী খুঁজছি?"
**Key (K)** — "আমার কাছে কী আছে?"
**Value (V)** — "আমি কী information দিতে পারি?"

প্রতিটা word এর Q, K, V calculate হয়। তারপর Q আর K এর dot product দিয়ে attention score বের হয়, সেই score দিয়ে V কে weight করে final output আসে।

এটা ভাবো এভাবে — তুমি একটা room এ ঢুকলে (Query), room এর সবাই তোমাকে tag দেখালো (Key), তুমি relevant tag দেখে ওদের কাছে গেলে information নিতে (Value)।

### Multi-Head Attention

একটা attention head একটা relationship ধরে। কিন্তু language এ অনেক রকম relationship আছে — grammatical, semantic, positional। তাই Transformer **multiple attention heads** চালায় parallel এ। প্রতিটা head আলাদা relationship শেখে। তারপর সবার output merge হয়।

ঠিক যেমন একটা meeting এ একজন finance দেখে, একজন tech দেখে, একজন legal দেখে — শেষে সবাই মিলে decision নেয়।

---

## Transformer এর দুইটা Part

Original Transformer architecture এ দুইটা part ছিলো:

### Encoder
- Input text পড়ে
- পুরো context বোঝে
- Internal representation তৈরি করে
- **Bidirectional** — বাম থেকে ডান আর ডান থেকে বাম দুই দিক থেকে দেখে

### Decoder
- Output generate করে
- Encoder এর representation ব্যবহার করে
- **Auto-regressive** — একটা word generate করে, তারপর সেটা ব্যবহার করে পরের word generate করে
- **Causal masking** — শুধু আগের words দেখতে পারে, ভবিষ্যতের words দেখতে পারে না

কিন্তু interesting ব্যাপার হলো — পরবর্তীতে সবাই full encoder-decoder ব্যবহার করেনি। কেউ শুধু encoder নিয়েছে, কেউ শুধু decoder, কেউ দুটোই। এখান থেকেই different model types এসেছে।

---

## তিন ধরনের Transformer Model

### ১. Encoder-Only — BERT Family

**BERT (Google)** — শুধু encoder ব্যবহার করে।

কী করে? পুরো sentence একবারে পড়ে, context বোঝে। কিন্তু নিজে text generate করে না।

কোথায় কাজে লাগে?
- Text classification ("এই review positive নাকি negative?")
- Sentiment analysis
- Named Entity Recognition ("Dhaka" একটা location)
- Embedding generation (text কে vector এ convert)
- Question answering

ভাবো — BERT হলো ভালো listener। সে বোঝে অনেক ভালো, কিন্তু নিজে বেশি কথা বলে না।

### ২. Decoder-Only — GPT, LLaMA Family

**GPT (OpenAI), LLaMA (Meta), Mistral** — শুধু decoder ব্যবহার করে।

কী করে? Next token predict করে। তুমি কিছু লিখলে, সে তার পরে কী আসা উচিত সেটা generate করে। একটার পর একটা token — এভাবে পুরো response তৈরি হয়।

কোথায় কাজে লাগে?
- Chatbot (ChatGPT, Claude)
- Code generation
- Story/content writing
- Reasoning

Causal masking ব্যবহার করে — মানে একটা word generate করার সময় সে শুধু আগের words দেখতে পারে, পরের words দেখতে পারে না। এটা logical — তুমি গল্প লেখার সময়ও পরের line জানো না, আগের lines দেখে লেখো।

ভাবো — GPT হলো ভালো speaker। সে কথা বলতে পারে অনেক ভালো, generate করতে পারে।

### ৩. Encoder-Decoder — T5 Family

**T5 (Google)** — দুটোই ব্যবহার করে। Encoder input বোঝে, Decoder output generate করে।

কোথায় কাজে লাগে?
- Translation ("বাংলা থেকে ইংরেজি")
- Summarization
- Text rewriting
- Any input → output transformation

ভাবো — T5 হলো ভালো translator। সে একটা জিনিস বুঝে আরেকটা জিনিসে convert করতে পারে।

---

## একনজরে: Model Families

| Model | Architecture | কী ভালো পারে | কী পারে না ভালো |
|-------|-------------|-------------|----------------|
| BERT | Encoder-only | Understanding, classification | Text generation |
| GPT | Decoder-only | Text generation, chat | Embedding (originally) |
| LLaMA | Decoder-only | Generation, reasoning | Same as GPT |
| T5 | Encoder-Decoder | Translation, transformation | Pure generation |

---

## Model বানাতে হলে কি Transformer ই লাগবে?

না। Transformer mandatory না।

**CNN** — image processing এ এখনো ব্যবহার হয়
**XGBoost / Random Forest** — tabular data তে এখনো Transformer এর চেয়ে ভালো
**RNN/LSTM** — খুব small dataset বা embedded system এ কাজে লাগে
**Classical ML** — simple classification, regression এ deep learning এর দরকার নাই

কিন্তু NLP/LLM বানাতে হলে? হ্যাঁ, Transformer ই standard এখন। কারণ অন্য কোনো architecture এই scale এ এতো ভালো কাজ করে না।

Rule of thumb: **Architecture follow করবে problem কে, hype কে না।** তুমি bulldozer দিয়ে ঘড়ি ঠিক করো না।

```
Timeline of dominance:
Images:    MLP → CNN → Vision Transformer (ViT)
Text:      RNN → LSTM → Transformer (GPT, BERT)
Audio:     RNN → Transformer (Whisper)
Tabular:   Still XGBoost/Random Forest wins often
```

---

## Architecture vs Framework vs Model vs Weights

এইটা একটা confusion যেটা প্রায় সবাই face করে। চলো permanently clear করি।

**Architecture** = Mathematical design। Blueprint। Transformer হলো architecture। এটা define করে Self-Attention কিভাবে কাজ করবে, কতো layers থাকবে, কিভাবে data flow হবে।

**Framework / Library** = Software code যেটা architecture implement করে। Hugging Face এর `transformers` library হলো framework। এটা তোমাকে ৩ লাইনে model load করতে দেয়।

**Model** = Architecture এর specific variant। GPT, BERT, LLaMA — এরা সবাই Transformer architecture ব্যবহার করে, কিন্তু different configuration এ। ঠিক যেমন Toyota আর BMW দুটোই "car" architecture, কিন্তু different design।

**Weights** = Trained parameters। Model train হওয়ার পরে যেই learned numbers থাকে। `bert-base-uncased` বা `llama-3-8b` — এগুলো specific weight files। এটাই actual intelligence।

```
Architecture (Transformer — math design)
        ↓
Framework (Hugging Face transformers — library)
        ↓
Model type (GPT / BERT / T5 / LLaMA)
        ↓
Pretrained Weights (actual trained intelligence)
```

Developer analogy:

```
TCP/IP              = Architecture concept
Node.js             = Framework implementation
Express App         = Specific server design
Deployed server     = Trained weights (actual running system)
```

আরেকটা:

```
React               = Architecture idea
Next.js             = Framework
তোমার dashboard app = Specific model
Production deploy   = Trained weights
```

---

## এই Chapter এ কী শিখলে?

তুমি জানলে Transformer আসার আগে কী ছিলো আর কেন replace হলো। Self-Attention কিভাবে কাজ করে — Q, K, V concept। Transformer এর Encoder আর Decoder part। তিন ধরনের model — Encoder-only (BERT), Decoder-only (GPT/LLaMA), Encoder-Decoder (T5)। Transformer mandatory না — problem অনুযায়ী architecture choose করতে হয়। আর Architecture, Framework, Model, Weights — চারটা আলাদা layer।

পরের chapter এ আমরা specific model types নিয়ে আরও deep যাবো — LLM, VLM, Diffusion Model, Token, Embedding, Temperature, Prompt Engineering। এবার আরও practical হবে।
