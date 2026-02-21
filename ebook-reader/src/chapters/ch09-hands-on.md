# Chapter 9: Hands-on — Model Download, Run, API Serve, Fine-tune

> "যথেষ্ট theory হয়েছে। এখন হাতে কলমে — model download করো, চালাও, API বানাও।"

---

## এই Chapter আলাদা

এতক্ষণ তুমি concept শিখেছো — tools, data pipeline, transformer, model types, runtime, Hugging Face, production reality। এই chapter এ কোনো theory নাই। শুধু code। শুধু করা।

প্রতিটা section এ একটা task আছে। তুমি follow করো, run করো, output দেখো। ভাঙো, ঠিক করো। এভাবেই শেখা হয়।

---

## Part 1: Setup — ৫ মিনিটে Ready হও

### Python environment

```bash
# Python 3.10+ লাগবে
python --version

# Virtual environment বানাও (recommended)
python -m venv ai-env
source ai-env/bin/activate  # Linux/Mac
# ai-env\Scripts\activate   # Windows

# Core packages install
pip install transformers torch huggingface_hub fastapi uvicorn
```

### Hugging Face CLI (optional — public model এ লাগবে না)

```bash
pip install huggingface_hub
huggingface-cli login
# Token paste করো: Settings → Access Tokens → Generate (write)
```

---

## Part 2: তোমার প্রথম Model Run — ৩ লাইনে

### Sentiment Analysis

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love building backend systems with NestJS")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

ব্যস। ৩ লাইন। Model automatically download হলো, inference হলো, result পেলে।

### Text Generation

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
result = generator("The future of software engineering is", max_new_tokens=50)
print(result[0]['generated_text'])
```

### Summarization

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
text = """
Hugging Face is a company that provides tools for building, training and deploying 
machine learning models. Their transformers library has become the standard way to 
work with large language models. The platform hosts hundreds of thousands of 
pre-trained models that developers can use in their applications.
"""
result = summarizer(text, max_length=50, min_length=20)
print(result[0]['summary_text'])
```

### Translation

```python
from transformers import pipeline

translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
result = translator("Hello, how are you?")
print(result[0]['translation_text'])
# Bonjour, comment allez-vous?
```

---

## Part 3: Bigger Model — Llama চালাও Locally

### Ollama দিয়ে (সবচেয়ে সহজ)

```bash
# Ollama install করো (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Model download + run
ollama run llama3.2:1b

# Chat শুরু হবে terminal এ
>>> What is a REST API?
```

1B model — CPU তেও চলবে। GPU দরকার নাই।

### Python দিয়ে (Transformers library)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

prompt = "Explain what a database index is in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Part 4: Embedding — Semantic Search বানাও

এইটা তোমার product এ সবচেয়ে আগে কাজে লাগবে।

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Embedding model load
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Documents
docs = [
    "NestJS is a framework for building server-side applications",
    "React is a library for building user interfaces",
    "PostgreSQL is a powerful relational database",
    "Docker containers package applications with dependencies",
]

# Query
query = "How to build a backend API?"
query_emb = get_embedding(query)

# Similarity calculate
import numpy as np

for doc in docs:
    doc_emb = get_embedding(doc)
    similarity = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
    print(f"{similarity:.4f} → {doc}")

# NestJS wala document er similarity সবচেয়ে বেশি হবে!
```

---

## Part 5: FastAPI দিয়ে Model কে REST API বানাও

এইটা তোমার backend experience directly কাজে লাগবে। Model কে একটা API endpoint হিসেবে serve করো।

```python
# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
classifier = pipeline("sentiment-analysis")
generator = pipeline("text-generation", model="gpt2")

class TextInput(BaseModel):
    text: str

class GenerateInput(BaseModel):
    prompt: str
    max_tokens: int = 50

@app.post("/sentiment")
def analyze_sentiment(input: TextInput):
    result = classifier(input.text)
    return {"label": result[0]["label"], "score": result[0]["score"]}

@app.post("/generate")
def generate_text(input: GenerateInput):
    result = generator(input.prompt, max_new_tokens=input.max_tokens)
    return {"generated_text": result[0]["generated_text"]}

@app.get("/health")
def health():
    return {"status": "ok"}
```

Run করো:

```bash
uvicorn app:app --reload --port 8000
```

Test করো:

```bash
# Sentiment
curl -X POST http://localhost:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "I love coding in TypeScript"}'

# Generate
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The best database for startups is", "max_tokens": 30}'
```

ব্যস! তোমার কাছে এখন একটা AI-powered REST API আছে। NestJS developer হিসেবে — এইটা তোমার comfort zone। Model serving মানে তো backend engineering ই।

---

## Part 6: Simple Fine-tune — নিজের Data দিয়ে

এইটা একটু advanced কিন্তু doable। Google Colab (free tier) এও চলবে।

### Data Prepare

```python
# training_data.json
data = [
    {"instruction": "What is NestJS?", "output": "NestJS is a progressive Node.js framework for building efficient server-side applications."},
    {"instruction": "What is PostgreSQL?", "output": "PostgreSQL is an advanced open-source relational database management system."},
    {"instruction": "What is Docker?", "output": "Docker is a platform for building, shipping, and running applications in containers."},
    # ... আরও ১০০-১০০০ examples
]
```

### LoRA Fine-tune (QLoRA — কম GPU তে চলে)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

# Base model load (4-bit quantized)
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # QLoRA — কম memory লাগবে
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# LoRA config — শুধু কিছু part train হবে
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Dataset
dataset = Dataset.from_list([
    {"text": f"### Instruction: {d['instruction']}\n### Response: {d['output']}"}
    for d in data
])

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
    ),
)
trainer.train()

# Save + push
model.save_pretrained("./my-fine-tuned-model")
# model.push_to_hub("tomar-username/my-model")  # HF তে upload
```

---

## Part 7: Hugging Face এ Upload

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo("tomar-username/swe-assistant-v1", repo_type="model")
api.upload_folder(
    folder_path="./my-fine-tuned-model",
    repo_id="tomar-username/swe-assistant-v1",
)
print("Model uploaded! 🎉")
```

এখন যে কেউ তোমার model use করতে পারবে:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("tomar-username/swe-assistant-v1")
```

---

## Full Workflow — একনজরে

```
Step 1: pip install transformers torch
Step 2: pipeline() দিয়ে model run করো (৩ লাইন)
Step 3: Different tasks try করো (sentiment, generation, embedding)
Step 4: FastAPI দিয়ে REST API বানাও
Step 5: নিজের data দিয়ে LoRA fine-tune করো
Step 6: Hugging Face এ upload করো
Step 7: Share করো, iterate করো
```

---

## PocketSchool Course Flow Suggestion

এই chapter এর content কে ৪টা class এ ভাগ করতে পারো:

**Class 1:** HF setup + pipeline() দিয়ে ৫টা task run (sentiment, generation, summarization, translation, QA)

**Class 2:** Embedding + semantic search বানাও। pgvector intro।

**Class 3:** FastAPI দিয়ে model serve করো REST API হিসেবে। Students এর "aha moment" — "AI engineering মানে backend engineering এর extension!"

**Class 4:** Fine-tune basics। LoRA দিয়ে small model fine-tune করো নিজের data দিয়ে।

1B model দিয়ে start করো সবসময়। GPU ছাড়াই CPU তে চলবে। Students এর laptop এও run করবে। পরে bigger model দেখাও।

---

## এই Chapter এ কী শিখলে?

তুমি হাতে কলমে করলে — model download, run, embedding, API serve, fine-tune, upload। এইটাই ছিলো পুরো বইয়ের লক্ষ্য — theory বোঝো, তারপর করো।

তোমার AI journey শুরু হয়ে গেছে। এখন iterate করো, build করো, break করো, fix করো। Happy building! 🚀
