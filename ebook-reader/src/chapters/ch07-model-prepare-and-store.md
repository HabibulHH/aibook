# Chapter 7: Model Prepare করা এবং Hugging Face এ Store করা

> "Model তো বুঝলাম আছে। কিন্তু এই জিনিসগুলো আসে কোথা থেকে? কিভাবে তৈরি হয়? আর কোথায় রাখা হয়?"

---

## Model "Prepare" করা বলতে আসলে কী বোঝায়?

আগের chapters এ তুমি Transformer architecture, model types, runtime, আর Hugging Face ecosystem বুঝেছো। এখন প্রশ্ন হলো — এই model গুলো কিভাবে তৈরি হয়?

চলো সোজা ভাষায় বলি। একটা AI model তৈরি করতে মূলত ৩টা ধাপ আছে। এবং তুমি একজন ডেভেলপার হিসেবে এই ৩টা ধাপের মধ্যে ২টা তে directly involve হতে পারো। তৃতীয়টাতে তোমার দরকার অনেক টাকা আর GPU — সেটা OpenAI, Google, Meta রা করে। কিন্তু বাকি ২টা? সেটা তুমিও পারবে।

---

## ধাপ ১: Pre-training (এইটা বড় ছেলেদের কাজ)

এইটা হলো সেই phase যেখানে একটা model কে পুরো internet এর data খাওয়ানো হয়। Wikipedia, books, code repositories, websites — সব। এই phase এ model শেখে ভাষা কিভাবে কাজ করে, কোন word এর পরে কোন word আসতে পারে, code এর pattern কী, ইত্যাদি।

এইটা করতে লাগে হাজার হাজার GPU, কোটি কোটি টাকা, এবং মাসের পর মাস সময়। GPT-4 তৈরি করতে OpenAI এর কোটি কোটি ডলার খরচ হয়েছে। Llama তৈরি করতে Meta এর হাজার হাজার GPU চলেছে।

তোমার এই ধাপটা করতে হবে না। এইটা foundation companies করে দিয়েছে। তোমার কাজ হলো এই foundation এর উপর বিল্ড করা।

---

## ধাপ ২: Fine-tuning (এইটা তুমি পারবে!)

এখন মজার পার্ট। ধরো Meta একটা Llama model release করেছে। সেটা general purpose — মানে সব কিছু একটু একটু পারে। কিন্তু তুমি চাও এই model টা specifically বাংলায় ভালো কাজ করুক। বা তুমি চাও এটা medical questions এ expert হোক। বা তোমার company এর customer support এর tone এ কথা বলুক।

এইটা হলো fine-tuning। তুমি একটা existing model নিচ্ছো, এবং তোমার নিজের data দিয়ে সেটাকে আরও specific করে তুলছো।

কিভাবে? ধরো তোমার কাছে ১০,০০০ টা বাংলা প্রশ্ন-উত্তর আছে। তুমি সেগুলো একটা নির্দিষ্ট format এ সাজাবে:

```json
{
  "instruction": "বাংলাদেশের রাজধানী কী?",
  "input": "",
  "output": "বাংলাদেশের রাজধানী ঢাকা।"
}
```

এরকম হাজার হাজার example দিয়ে তুমি model কে train করবে। এটাকে বলে **Supervised Fine-Tuning (SFT)**।

Fine-tuning এর জন্য তোমার দরকার হবে GPU। তুমি নিজের machine এ করতে পারো (যদি ভালো GPU থাকে), অথবা cloud এ করতে পারো — Google Colab, AWS SageMaker, Lambda Labs, RunPod এসব দিয়ে।

Tools যেগুলো fine-tuning এ কাজে লাগে:

**Hugging Face Transformers** — সবচেয়ে popular library। Python এ `from transformers import AutoModelForCausalLM` লিখে model load করো, তোমার data দাও, train করো।

**LoRA / QLoRA** — full model train করতে অনেক GPU লাগে। LoRA একটা technique যেটা model এর শুধু কিছু part train করে, বাকিটা freeze রাখে। এতে অনেক কম GPU তে কাজ হয়ে যায়। QLoRA আরও efficient — 4-bit quantization সহ LoRA।

**Unsloth** — LoRA fine-tuning কে আরও fast করে দেয়। Free tier এ Google Colab এও চলে।

**Axolotl** — config file দিয়ে fine-tuning setup করা যায়। YAML এ define করো কোন model, কোন data, কোন method — বাকিটা সে করে দেবে।

---

## ধাপ ৩: RLHF / Alignment (Model কে ভদ্র বানানো)

এইটা হলো শেষ ধাপ। Fine-tune করার পরেও model অনেক সময় এমন কথা বলে যেটা ঠিক না — harmful content, biased answer, বা এলোমেলো response। এই ধাপে human feedback দিয়ে model কে শেখানো হয় কোনটা ভালো answer আর কোনটা খারাপ।

এইটাকে বলে **RLHF — Reinforcement Learning from Human Feedback**।

তুমি যদি production level model বানাও, তাহলে এই ধাপটা important। কিন্তু শুরুতে fine-tuning দিয়ে শুরু করলেই যথেষ্ট।

---

## Hugging Face — Model এর GitHub

এখন সবচেয়ে practical প্রশ্ন: model তৈরি করলে সেটা রাখবে কোথায়? শেয়ার করবে কিভাবে?

উত্তর হলো **Hugging Face**। এইটাকে ভাবো model এর GitHub। যেমন তুমি code রাখো GitHub এ, model রাখবে Hugging Face এ।

---

### Hugging Face কী কী করে?

**Model Hub** — হাজার হাজার pre-trained model এখানে আছে। Llama, Mistral, Falcon, Phi — সব পাবে। তুমি ডাউনলোড করতে পারো, নিজের project এ use করতে পারো।

**Dataset Hub** — model train করতে data লাগে। সেই data ও এখানে পাবে। Bengali NLP dataset থেকে শুরু করে medical data পর্যন্ত।

**Spaces** — model deploy করে demo দেখাতে চাও? Hugging Face Spaces এ Gradio বা Streamlit app deploy করতে পারো free তে।

**Inference API** — model ডাউনলোড না করেই API call করে use করতে পারো।

---

### Hugging Face এ Account তৈরি করা

প্রথমে [huggingface.co](https://huggingface.co) তে গিয়ে একটা account তৈরি করো। Free account এই অনেক কিছু করা যায়।

তারপর CLI install করো:

```bash
pip install huggingface_hub
```

Login করো:

```bash
huggingface-cli login
```

এটা তোমাকে একটা token চাইবে। Hugging Face এর Settings > Access Tokens এ গিয়ে একটা token generate করো (write permission দিয়ে), সেটা paste করো। ব্যস, setup done।

---

### Model Upload করা Hugging Face এ

ধরো তুমি একটা Llama model fine-tune করেছো বাংলা data দিয়ে। এখন সেটা Hugging Face এ upload করতে চাও। দুইটা উপায় আছে:

**উপায় ১: Python দিয়ে (সবচেয়ে common)**

```python
from huggingface_hub import HfApi

api = HfApi()

# নতুন repo তৈরি করো
api.create_repo(repo_id="tomar-username/bangla-llama-7b", repo_type="model")

# model files upload করো
api.upload_folder(
    folder_path="./my-fine-tuned-model",   # তোমার local model folder
    repo_id="tomar-username/bangla-llama-7b",
    repo_type="model",
)
```

**উপায় ২: CLI দিয়ে**

```bash
# repo তৈরি করো
huggingface-cli repo create bangla-llama-7b

# git clone করো
git clone https://huggingface.co/tomar-username/bangla-llama-7b
cd bangla-llama-7b

# model files copy করো এই folder এ
cp -r ../my-fine-tuned-model/* .

# git lfs setup (বড় files এর জন্য)
git lfs install
git lfs track "*.bin"
git lfs track "*.safetensors"

# push করো
git add .
git commit -m "Upload fine-tuned Bangla Llama model"
git push
```

**উপায় ৩: Transformers library দিয়ে directly**

এইটা সবচেয়ে সহজ যদি তুমি Hugging Face Transformers দিয়ে train করে থাকো:

```python
# training শেষে
model.push_to_hub("tomar-username/bangla-llama-7b")
tokenizer.push_to_hub("tomar-username/bangla-llama-7b")
```

ব্যস! দুই লাইনে model Hugging Face এ upload হয়ে গেলো।

---

### Model Card লেখা — এইটা Skip করো না

Hugging Face এ model upload করলে একটা `README.md` file থাকে — এটাকে বলে Model Card। এটা হলো তোমার model এর documentation। এখানে লিখবে:

- **Model কী করে** — কোন task এর জন্য, কোন ভাষায়
- **Base model কোনটা** — Llama 3, Mistral, ইত্যাদি
- **Training data** — কী data দিয়ে train করেছো
- **How to use** — কিভাবে load করে use করবে, code example সহ
- **Limitations** — model এর কোথায় দুর্বলতা আছে

Example:

```markdown
---
language: bn
license: mit
base_model: meta-llama/Llama-3-8B
tags:
  - bengali
  - fine-tuned
  - text-generation
---

# Bangla Llama 7B

Bengali language fine-tuned version of Llama 3 8B.

## Usage

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("tomar-username/bangla-llama-7b")
tokenizer = AutoTokenizer.from_pretrained("tomar-username/bangla-llama-7b")
```

---

### Dataset Upload করা

শুধু model না, training data ও Hugging Face এ রাখতে পারো:

```python
from datasets import Dataset
import pandas as pd

# তোমার data load করো
df = pd.read_csv("bangla_qa_dataset.csv")
dataset = Dataset.from_pandas(df)

# upload করো
dataset.push_to_hub("tomar-username/bangla-qa-10k")
```

এখন যে কেউ তোমার dataset use করতে পারবে:

```python
from datasets import load_dataset
dataset = load_dataset("tomar-username/bangla-qa-10k")
```

---

## Model File Format বোঝা

Hugging Face এ model upload করলে কিছু files থাকে। এগুলো বোঝা দরকার:

**config.json** — model এর architecture define করে। কতো layers, কতো attention heads, vocab size কতো — সব এখানে।

**tokenizer.json / tokenizer_config.json** — tokenizer এর configuration। Text কে কিভাবে token এ ভাগ করবে সেটা এখানে defined।

**model.safetensors** — এইটা হলো actual model weights। আগে `.bin` (PyTorch format) ব্যবহার হতো, এখন `.safetensors` হলো standard — বেশি safe এবং fast।

**generation_config.json** — generation এর default settings। Temperature, top_p, max tokens ইত্যাদি।

মোটামুটি এই files গুলো থাকলেই যে কেউ তোমার model load করে use করতে পারবে।

---

## Quantization — Model ছোট করা

এইটা একটা practical problem। ধরো তোমার model 7B parameters এর — মানে প্রায় ১৪ GB (FP16 format এ)। তোমার কাছে এতো GPU নাই।

Solution হলো **Quantization**। এটা model এর precision কমিয়ে size ছোট করে:

| Format | Size (7B model) | Quality | Use Case |
|--------|-----------------|---------|----------|
| FP16 | ~14 GB | Best | Training, high-end GPU |
| INT8 | ~7 GB | Very Good | Good GPU তে inference |
| INT4 (GPTQ/AWQ) | ~4 GB | Good | Consumer GPU তে চলে |
| GGUF (llama.cpp) | ~4-5 GB | Good | CPU তেও চলে! |

Hugging Face এ অনেক model এর quantized version পাবে। যেমন `TheBloke` নামের একজন user প্রায় সব popular model এর GPTQ আর GGUF version upload করে রাখতেন।

Quantize করতে চাইলে:

```python
# bitsandbytes দিয়ে 4-bit load
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
)
```

---

## Practical Workflow: শুরু থেকে শেষ

তাহলে পুরো workflow টা একসাথে দেখি:

**Step 1:** Hugging Face থেকে base model download করো (যেমন Llama 3 8B)

**Step 2:** তোমার dataset তৈরি করো (instruction + input + output format এ)

**Step 3:** LoRA/QLoRA দিয়ে fine-tune করো (Google Colab বা cloud GPU তে)

**Step 4:** Fine-tuned model টা Hugging Face এ upload করো (`push_to_hub`)

**Step 5:** Model Card লিখো — কী করে, কিভাবে use করবে

**Step 6:** শেয়ার করো! অন্যরা তোমার model download করে use করতে পারবে

---

## এই Chapter এ কী শিখলে?

আজকে তুমি জানলে model তৈরি হওয়ার ৩টা ধাপ — pre-training, fine-tuning, আর RLHF। তুমি জানলে Hugging Face হলো model এর GitHub — যেখানে model upload করা যায়, download করা যায়, share করা যায়। তুমি দেখলে কিভাবে Python দিয়ে, CLI দিয়ে, বা Transformers library দিয়ে model upload করতে হয়। আর quantization দিয়ে কিভাবে বড় model কে ছোট করে চালানো যায়।

পরের chapter এ আমরা ঢুকবো model এর ভেতরে — LLM কিভাবে কাজ করে, token কী জিনিস, temperature মানে কী, আর prompt engineering এর basics। সেটাও হবে একদম হাতে কলমে।

তোমার কাজ? আপাতত Hugging Face এ একটা account খোলো। ঘুরে দেখো কী কী model আছে। একটা model download করে locally চালানোর চেষ্টা করো Ollama দিয়ে। এইটাই তোমার homework।
