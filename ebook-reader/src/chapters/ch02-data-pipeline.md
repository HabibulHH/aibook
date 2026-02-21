# Chapter 2: Data → Model Pipeline — Model এ Data ঢোকার আগে কী হয়?

> "একটা model যখন কিছু predict করে, তখন raw data সরাসরি ঢোকে? নাকি আগে কিছু হয়?"

---

## হ্যাঁ, আগে অনেক কিছু হয়

আগের chapter এ তুমি দেখলে কোন কোন AI tools আছে আর সেগুলোর পেছনে কোন model কাজ করছে। এখন প্রশ্ন হলো — এই model গুলোর কাছে data কিভাবে পৌঁছায়?

সোজা উত্তর: raw data সরাসরি model এ যায় না। আগে একটা পুরো pipeline আছে। সেই pipeline এ data clean হয়, transform হয়, prepare হয় — তারপর model এ ঢোকে।

```
Raw Data → Clean → Transform → Features → Model
```

এই chapter এ আমরা এই pipeline টা ভাঙবো। আর দেখবো traditional ML আর Transformer (GPT/BERT type model) এ এই pipeline কিভাবে আলাদা।

---

## Data Collection — শুরুটা কোথা থেকে?

প্রথম ধাপ হলো data collect করা। তোমার data source যেকোনো কিছু হতে পারে — database, API, CSV file, logs, user activity, web scraping। এইটা তোমার problem এর উপর depend করে।

ধরো তুমি একটা sentiment analysis model বানাতে চাও বাংলায়। তোমার data হতে পারে Facebook comments, product reviews, news articles। এগুলো collect করাই প্রথম কাজ।

কিন্তু collect করলেই হলো না। Real world data দেখতে সুন্দর না — dirty, messy, incomplete।

---

## Data Cleaning — ৮০% Effort এখানেই যায়

এইটা শুনতে boring লাগবে, কিন্তু সত্যি কথা হলো — ML এর ৮০% effort cleaning এ যায়। Model বানানো মজার পার্ট, কিন্তু সেই পার্টে পৌঁছানোর আগে তোমাকে data ঠিক করতে হবে।

কী কী ঠিক করতে হয়?

**Missing values** — কিছু row তে data নাই। হয় সেই row বাদ দাও, হয় কোনো value দিয়ে fill করো (mean, median, বা intelligent guess)।

**Duplicates** — একই data যদি ১০ বার থাকে, model মনে করবে সেই pattern ১০ গুণ বেশি important। তাই duplicate সরাও।

**Outliers** — ধরো তোমার dataset এ সবার salary 30K-80K range এ, কিন্তু একজনের 99,99,999। এই outlier model কে confuse করবে।

**Invalid entries** — ভুল format, typo, impossible values (যেমন age: -5)। এগুলো fix বা remove করো।

ভাবো এভাবে — তুমি রান্না করবে, কিন্তু চাল ধোয়া হয়নি, সবজিতে পোকা আছে, মশলা expire হয়ে গেছে। আগে এগুলো ঠিক না করলে রান্না যতোই ভালো করো, খাবার ভালো হবে না। Data cleaning ঠিক এই কাজটাই।

---

## Feature Engineering — ৮০% Impact এখানে পড়ে

এইটা arguably সবচেয়ে important step। Feature Engineering মানে হলো — raw data থেকে meaningful features বানানো যেটা model কে ভালোভাবে শিখতে সাহায্য করবে।

Example দিই। ধরো তুমি একটা e-commerce site এর জন্য model বানাচ্ছো — "এই customer কি আবার কিনবে?"

তুমি যদি শুধু `price` দাও model কে — result মোটামুটি হবে।

কিন্তু তুমি যদি দাও:
- `price`
- `discount_percentage`
- `days_since_last_purchase`
- `total_orders_last_6_months`
- `average_order_value`

Result dramatically ভালো হবে। কারণ তুমি model কে বেশি context দিচ্ছো।

এইটাই Feature Engineering — তুমি model কে "কী দেখাচ্ছো" সেটা determine করে model কতো ভালো perform করবে। Garbage in, garbage out — এই কথাটা ML এ সবচেয়ে বেশি সত্যি।

---

## Data Transformation — Number এ Convert করা

ML model number বোঝে। Text, category, image — সরাসরি বোঝে না। তাই সব কিছু number এ convert করতে হয়। এইটাকে বলে transformation বা encoding।

### Categorical Data → Numbers

ধরো তোমার data তে একটা column আছে "color" — values হলো "red", "blue", "green"। Model এইটা directly বুঝবে না। দুইটা উপায় আছে:

**Label Encoding** — red=0, blue=1, green=2। Simple, কিন্তু একটা problem — model মনে করতে পারে green(2) > blue(1) > red(0), যেটা মানে হয় না।

**One-Hot Encoding** — প্রতিটা value এর জন্য আলাদা column:

```
red   → [1, 0, 0]
blue  → [0, 1, 0]
green → [0, 0, 1]
```

এইটা better — কোনো false ordering নাই।

### Numerical Data → Scaling

ধরো তোমার দুইটা feature আছে — `age` (20-60 range) আর `salary` (20000-100000 range)। Salary এর number অনেক বড়, তাই model মনে করবে salary বেশি important — শুধু range বড় বলে, actual importance এর কারণে না।

Solution হলো Scaling:

**StandardScaler** — mean=0, std=1 করে দেয়। সবচেয়ে common।

**MinMaxScaler** — সব value কে 0-1 range এ নিয়ে আসে।

---

## Train/Test Split — পরীক্ষার খাতা আলাদা রাখো

Model কে train করার পর তোমাকে জানতে হবে — সে কি সত্যিই শিখেছে, নাকি শুধু মুখস্থ করেছে?

এজন্য data split করো:

**Training set (70-80%)** — এই data দিয়ে model শেখে।

**Validation set (10-15%)** — training এর সময় tune করার জন্য। Hyperparameter adjust করো এই set দেখে।

**Test set (10-15%)** — final evaluation। model এই data আগে কখনো দেখেনি। এইটা হলো real exam।

Analogy: তুমি practice problem solve করো (training), mock test দাও (validation), আর final exam দাও (test)। নিজের homework এ ভালো করলেই হবে না — exam এও ভালো করতে হবে। Model এর ক্ষেত্রেও একই কথা।

---

## এখন পর্যন্ত যেটা দেখলে — এটা Traditional ML

উপরের পুরো pipeline টা হলো traditional ML এর জন্য — যেখানে তুমি XGBoost, Random Forest, SVM এসব ব্যবহার করো।

```
Traditional ML Pipeline:
Raw Data 
  → Cleaning (80% of effort)
  → Feature Engineering (80% of impact)
  → Encoding + Scaling
  → Train/Val/Test Split
  → Model Training (XGBoost, SVM, etc.)
```

কিন্তু Transformer based model (GPT, BERT, LLaMA) এর preprocessing সম্পূর্ণ আলাদা। চলো দেখি।

---

## Transformer এর Pipeline — সম্পূর্ণ আলাদা গল্প

Transformer based model এ তুমি manually feature engineering করো না। Tokenization আর Embedding — এই দুইটা step করলেই হয়। বাকিটা model নিজে শেখে।

```
Transformer Pipeline:
Raw Text → Tokenization → Token IDs → Embedding → Transformer Model
```

### Tokenization — Text ভাঙার কারিগর

Transformer text সরাসরি পড়ে না। সে token পড়ে। Tokenization হলো text কে ছোট ছোট টুকরোতে ভাঙা।

```
"I love coding" → ["I", " love", " cod", "ing"]
```

দেখো, "coding" দুই টুকরো হয়ে গেলো — "cod" আর "ing"। এইটা করে **BPE (Byte Pair Encoding)** — GPT এইটা ব্যবহার করে। Tokenizer commonly used patterns শিখে রাখে, তারপর সেই অনুযায়ী ভাঙে।

### Token → ID Mapping

প্রতিটা token এর একটা numerical ID থাকে vocabulary তে:

```
"I"     → 101
" love" → 4521
" cod"  → 3876
"ing"   → 2341
```

### Embedding — ID থেকে Rich Vector

এখন এই ID গুলো model এর Embedding layer এ ঢোকে। সেখানে প্রতিটা ID একটা dense vector এ convert হয় — হয়তো ৭৬৮ বা ১০২৪ টা number এর list:

```
101  → [0.2, 0.5, -0.1, 0.8, ...]
4521 → [0.8, 0.1, 0.3, -0.2, ...]
```

এই vectors semantic meaning carry করে। "love" আর "like" এর vector কাছাকাছি হবে, "love" আর "table" এর vector অনেক দূরে হবে।

### Positional Encoding — কে কোথায় আছে?

Transformer এ একটা মজার সমস্যা আছে। RNN/LSTM word গুলো একটার পর একটা পড়তো, তাই position automatically বোঝা যেতো। কিন্তু Transformer পুরো sequence একবারে দেখে — তাই সে জানে না "I" প্রথমে আছে নাকি শেষে।

এজন্য Positional Encoding add হয় — embedding এর সাথে position information যোগ করা হয়।

```
"I love coding"
  → Tokenizer: [101, 4521, 3876, 2341]
  → Embedding: [[0.2, 0.5, ...], [0.8, 0.1, ...], ...]
  → + Positional Encoding
  → Transformer Layers (Self-Attention + FFN)
  → Output
```

---

## বড় পার্থক্যটা কী?

Traditional ML তে **তুমি** feature বানাও — manually decide করো কোনটা important, কোনটা না। Model শুধু তোমার বানানো features থেকে pattern শেখে।

Transformer এ **model নিজেই** feature শিখে — তুমি শুধু raw text দাও tokenize করে, বাকি কাজ Attention mechanism আর Transformer layers করে। এইটাই key difference।

এই কারণেই Transformer এতো powerful — সে এমন patterns খুঁজে পায় যেটা তুমি manually feature engineering করে কখনো ধরতে পারতে না।

---

## একনজরে: দুই Pipeline

| Aspect | Traditional ML | Transformer |
|--------|---------------|-------------|
| Preprocessing | Manual, heavy | Tokenization only |
| Feature Engineering | তুমি করো | Model নিজে শেখে |
| Input format | Numbers (encoded) | Token IDs + Embeddings |
| Effort distribution | Cleaning 80%, Feature Eng 80% | Data quality + Tokenizer choice |
| Examples | XGBoost, SVM, Random Forest | GPT, BERT, LLaMA |

---

## এই Chapter এ কী শিখলে?

তুমি জানলে raw data সরাসরি model এ যায় না — একটা পুরো pipeline আছে। Data cleaning এ ৮০% effort যায়। Feature engineering এ ৮০% impact পড়ে। Encoding আর scaling ছাড়া ML model number বুঝতে পারে না। Train/test split না করলে model মুখস্থ করবে, শিখবে না। আর Transformer এর pipeline সম্পূর্ণ আলাদা — tokenization আর embedding ই তার preprocessing।

পরের chapter এ আমরা ঢুকবো Transformer architecture এর ভেতরে — এই জিনিসটা আসলে কী, কিভাবে কাজ করে, আর GPT, BERT, LLaMA কি একই জিনিস নাকি আলাদা।
