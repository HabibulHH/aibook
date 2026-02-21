# Chapter 6: Hugging Face — Model এর GitHub

> "Hugging Face বলে বলে সবাই — এইটা আসলে কী? npm এর মতো? GitHub এর মতো? কিভাবে কাজ করে?"

---

## Hugging Face আসলে কী?

সোজা কথায় — Hugging Face হলো AI/ML এর সবচেয়ে বড় open-source platform। তুমি যেমন code রাখো GitHub এ, model রাখবে Hugging Face এ।

কিন্তু এটা শুধু model hosting platform না — এটা একটা পুরো ecosystem। Library আছে, model hub আছে, dataset hub আছে, demo hosting আছে, inference API আছে।

শুরু হয়েছিলো একটা NLP library হিসেবে। এখন হয়ে গেছে ML এর GitHub + npm + DockerHub — সব একসাথে।

---

## Hugging Face এ কী কী আছে?

### `transformers` library — Main Library

এইটা Hugging Face এর core product। Python library যেটা দিয়ে তুমি ৩ লাইনে model load করতে পারো:

```python
from transformers import pipeline
pipe = pipeline("sentiment-analysis")
result = pipe("I love backend engineering")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

GPT, BERT, T5, LLaMA, Mistral — সব model এই library support করে।

### `diffusers` library — Image Generation

Stable Diffusion, Flux এর মতো diffusion model এর জন্য আলাদা library:

```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
image = pipe("a cat astronaut in space").images[0]
```

### `timm` library — Image Models

PyTorch Image Models। CNN based models ও আছে — ResNet, EfficientNet ইত্যাদি।

### Model Hub — হাজার হাজার Model

[huggingface.co/models](https://huggingface.co/models) — এখানে লক্ষ লক্ষ pre-trained model আছে। LLM, image model, audio model, embedding model — সব পাবে। Download করো, use করো।

### Dataset Hub — Training Data

[huggingface.co/datasets](https://huggingface.co/datasets) — model train করতে data লাগে। সেই data ও এখানে পাবে। Bengali NLP dataset থেকে medical data পর্যন্ত।

### Spaces — Live Demo

model এর live demo দেখাতে চাও? Gradio বা Streamlit app deploy করো free তে Hugging Face Spaces এ।

### Inference API — No Deploy Needed

Model download না করেই API call করে use করতে পারো:

```python
import requests
response = requests.post(
    "https://api-inference.huggingface.co/models/bert-base-uncased",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={"inputs": "The capital of Bangladesh is [MASK]."}
)
```

---

## এইটা npm এর মতো না কেন?

এই প্রশ্নটা অনেকেই করে। দুটোই তো package/model download করতে দেয়, তাহলে আলাদা কেন?

**npm distributes code** — lightweight, deterministic, কয়েক MB। `npm install express` করলে কিছু JavaScript file আসে।

**Hugging Face distributes trained models** — 2GB থেকে 200GB+। এগুলো code না, massive weight files (tensors)।

আরও কিছু fundamental পার্থক্য:

**GPU dependency** — তুমি `npm install gpt` করে Node process এ চালাতে পারো না। Model এর GPU দরকার efficient inference এর জন্য।

**Different runtimes** — npm package Node.js এ চলে। ML model এর জন্য PyTorch, TensorFlow, ONNX — আলাদা runtime লাগে (আগের chapter এ দেখেছো)।

**Git LFS** — npm small files handle করে। Hugging Face Git LFS (Large File Storage) ব্যবহার করে multi-GB binary files manage করতে।

**Dataset versioning** — npm এ dataset version control এর concept নাই। Hugging Face এ dataset versioning built-in।

সোজা কথায়: **npm = application code এর জন্য, Hugging Face = trained AI artifacts এর জন্য।** তুমি একই project এ দুটোই ব্যবহার করবে — npm তোমার app dependencies এর জন্য, Hugging Face তোমার AI model এর জন্য।

---

## `transformers` library আর Transformer architecture কি same?

না! এইটা common confusion।

**Transformer** = architecture idea। ২০১৭ সালের research paper। Mathematical concept।

**`transformers`** = Python library। Hugging Face বানিয়েছে। এই library Transformer architecture implement করে।

ঠিক যেমন:
- React = UI architecture idea
- Next.js = framework যেটা সেই idea implement করে

বা:
- REST = architectural style
- Express.js = library যেটা সেই style implement করে

তাই কেউ বললে "transformers use করো" — সে library এর কথা বলছে, architecture এর কথা না।

---

## Hugging Face কি সব ধরনের Model Support করে?

সংক্ষেপে — সব না, কিন্তু বেশিরভাগ।

**Strongly supported:**
- Transformer variants (GPT, BERT, LLaMA, Mistral, ViT, Whisper)
- Diffusion models (`diffusers` library দিয়ে)
- কিছু CNN models (`timm` দিয়ে)

**Hub এ আছে কিন্তু native library support নেই:**
- XGBoost, LightGBM, scikit-learn models
- তুমি upload করতে পারো, কিন্তু `from transformers import` দিয়ে load করা যাবে না

**Focus না:**
- Classical ML algorithms
- Custom non-neural architectures

মনে রাখো — Hugging Face শুরু হয়েছিলো NLP Transformer library হিসেবে। এখন ধীরে ধীরে ML এর GitHub হয়ে যাচ্ছে — Hub এ তুমি যেকোনো model upload করতে পারো।

---

## `from transformers import pipeline` — Login লাগে?

এইটা practical question। উত্তর:

**Public model এ login লাগে না।** তুমি just code লিখলেই model automatically download হয়ে যাবে:

```python
from transformers import pipeline
pipe = pipeline("sentiment-analysis")  # auto download, no login
```

Under the hood কী হচ্ছে?

```
1. Hugging Face Hub এ HTTP request যায়
2. Model files download হয় (config.json, model.safetensors, tokenizer.json)
3. Local cache এ save হয়: ~/.cache/huggingface/hub/
4. Cache থেকে memory তে load হয়
5. Pipeline ready
```

দ্বিতীয়বার run করলে আবার download করবে না — cache থেকে load হবে।

**Login কখন লাগে?**
- **Gated models** — LLaMA এর মতো model যেখানে Meta এর form fill করতে হয় access পেতে
- **Private repos** — তোমার নিজের private model
- **Upload করতে হলে** — model push করতে login mandatory

```bash
pip install huggingface_hub
huggingface-cli login  # token paste করো
```

Token পাবে: Hugging Face Settings → Access Tokens → Generate (write permission দিয়ে)

---

## AI/ML Developer হিসেবে কী Push করতে পারো?

Hugging Face শুধু download করার জায়গা না — তুমিও contribute করতে পারো:

**Fine-tuned Models** — তুমি Llama fine-tune করলে Bengali data দিয়ে? Push করো।

**LoRA/QLoRA Adapters** — Full model না, শুধু adapter weights। Size অনেক ছোট।

**Custom Datasets** — Bengali QA dataset, code review dataset, educational content — এগুলো community এর জন্য huge value।

**Tokenizers** — Bengali text এর জন্য custom tokenizer বানালে? Push করো।

**Quantized Models** — GGUF, GPTQ version বানালে — push করো।

**Spaces** — Demo app বানালে Gradio দিয়ে? Deploy করো Spaces এ।

```python
# Model push
model.push_to_hub("tomar-username/bangla-llama-7b")
tokenizer.push_to_hub("tomar-username/bangla-llama-7b")

# Dataset push
dataset.push_to_hub("tomar-username/bangla-qa-10k")
```

তোমার biggest opportunity: **Bengali/Bangla NLP space এ এখনো অনেক gap আছে।** তুমি dataset + model push করলে community impact + personal branding দুটোই হবে।

---

## Practical: ৩ লাইনে শুরু

Students দের জন্য সবচেয়ে সহজ start:

```python
# pip install transformers torch
from transformers import pipeline

# No login, no setup, auto download
pipe = pipeline("sentiment-analysis")
result = pipe("I love backend engineering")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

৩ লাইন। Login নাই। Setup নাই। Just pip install আর run। তোমার students instantly result দেখবে — এটাই best hook।

---

## এই Chapter এ কী শিখলে?

Hugging Face হলো ML এর GitHub — model, dataset, demo সব এখানে। npm আর Hugging Face fundamental ভাবে আলাদা কারণ model হলো massive binary files, code না। `transformers` library আর Transformer architecture same না — library architecture implement করে। Public model এ login লাগে না, gated model আর upload এ লাগে। আর তুমি নিজেও model, dataset, tokenizer, demo push করতে পারো।

পরের chapter এ আমরা দেখবো model কিভাবে prepare করে — pre-training, fine-tuning, RLHF, এবং Hugging Face এ কিভাবে upload আর store করে।
