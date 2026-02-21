# Chapter 5: Runtime — Model চালানোর Engine

> "Model download করলাম ঠিক আছে। কিন্তু এটা চলে কিসের উপর? Node.js যেমন JavaScript চালায়, model চালায় কে?"

---

## Runtime মানে কী?

তুমি একজন backend developer। তুমি জানো JavaScript চালাতে Node.js লাগে, Python চালাতে Python interpreter লাগে। একটা `.js` file নিজে নিজে কিছু না — Node.js ছাড়া সে useless।

AI model এর ক্ষেত্রেও ঠিক একই ব্যাপার। একটা model file (`.safetensors`, `.bin`, `.gguf`) হলো শুধু কিছু numbers — billions of weights। এই numbers কে execute করতে, matrix multiplication চালাতে, GPU তে load করতে একটা **runtime** লাগে।

Runtime = the engine that takes model weight files and actually runs the math on hardware.

---

## Training Time vs Inference Time

দুইটা আলাদা phase বুঝে নাও:

**Training Time** — model শিখছে। Weights update হচ্ছে। Expensive, slow, high compute। একবার হয় (বা occasionally)।

**Inference Time (Runtime)** — model already trained। তুমি input দিচ্ছো, output পাচ্ছো। Weights change হচ্ছে না। প্রতিটা user request এ এটা হয়।

তুমি যখন ChatGPT তে কিছু জিজ্ঞেস করো — সেই moment টা হলো inference/runtime। Model তখন run হচ্ছে, তোমার input process করছে, output generate করছে।

---

## বড় তিনটা Runtime

### ১. PyTorch — ML এর Node.js

**Meta (Facebook)** বানিয়েছে। এখন সবচেয়ে popular ML runtime।

কেন popular?
- Hugging Face এর বেশিরভাগ model PyTorch-native
- Dynamic computation graph — মানে normal Python এর মতো debug করতে পারো
- Research community এর default choice
- Biggest ecosystem — library, tutorial, community support সব বেশি

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
# PyTorch internally model এর math execute করছে
```

ভাবো — PyTorch হলো ML world এর Node.js। সবচেয়ে বেশি মানুষ ব্যবহার করে, সবচেয়ে বড় community, সবচেয়ে বেশি package support।

### ২. TensorFlow — ML এর Java/Spring

**Google** বানিয়েছে। আগে dominant ছিলো, এখন PyTorch এর কাছে হেরে গেছে research এ। কিন্তু production আর enterprise level এ এখনো ব্যবহার হয়।

কেন এখনো relevant?
- Google নিজে ব্যবহার করে
- TensorFlow Serving — production deployment এ mature
- TensorFlow Lite — mobile/edge devices এ model চালানো
- Google Cloud TPU support

ভাবো — TensorFlow হলো Java/Spring এর মতো। Enterprise-y, rigid, কিন্তু battle-tested। নতুন project এ কম ব্যবহার হয়, কিন্তু existing production system এ অনেক আছে।

### ৩. ONNX Runtime — Universal Bytecode

**Microsoft** বানিয়েছে। এইটা interesting concept — universal runtime।

তুমি PyTorch এ model train করো, তারপর ONNX format এ export করো, আর ONNX Runtime এ চালাও। কেন? কারণ ONNX inference এ optimized — CPU তেও efficiently চলে, GPU তেও।

```
PyTorch model (.pt) → Export → ONNX model (.onnx) → ONNX Runtime → Fast inference
```

ভাবো — Java bytecode এর মতো। Java source code compile হয়ে bytecode হয়, JVM যেকোনো platform এ চালায়। ONNX ও তাই — train যেখানে চাও করো, inference ONNX Runtime এ করো।

---

## LLM-Specific Runtimes

উপরের তিনটা general purpose। কিন্তু LLM চালানোর জন্য কিছু specialized runtime আছে যেগুলো অনেক important:

### llama.cpp — CPU তে LLM চালানো

এইটা একটা game-changer। আগে LLM চালাতে GPU mandatory ছিলো। llama.cpp দেখালো — quantized model (GGUF format) CPU তেও চলে! তোমার laptop এও LLM চলবে।

```bash
# llama.cpp দিয়ে model চালানো
./main -m llama-3-8b-Q4_K_M.gguf -p "What is REST API?"
```

Ollama internally llama.cpp ব্যবহার করে। তাই তুমি `ollama run llama3` করলে আসলে llama.cpp চলছে পেছনে।

### vLLM — Production LLM Serving

যখন তুমি production এ LLM serve করবে — মানে অনেক user একসাথে request করছে — তখন normal PyTorch দিয়ে চালালে slow হবে। vLLM specialized:

- **Continuous batching** — অনেক request একসাথে efficiently process করে
- **PagedAttention** — GPU memory efficiently ব্যবহার করে
- **High throughput** — normal PyTorch এর চেয়ে কয়েক গুণ fast

ভাবো — PyTorch হলো development server (nodemon), vLLM হলো production server (nginx + PM2)।

### TGI (Text Generation Inference) — Hugging Face এর

Hugging Face এর নিজের production serving solution। vLLM এর competitor। Docker image দিয়ে deploy করা যায়।

---

## Mobile / Edge Device Runtime

Model শুধু server এ চলে না। Phone, tablet, IoT device — এসবেও চলে। কিন্তু সেখানে আলাদা runtime লাগে কারণ resource limited।

### Core ML — Apple Devices

Apple এর নিজস্ব runtime। iPhone, iPad, Mac এর **Neural Engine** (dedicated ML chip) এ চলে।

```
PyTorch model → convert (coremltools) → Core ML model (.mlmodel) → iOS app
```

তুমি যদি Bengali speech-to-text model বানিয়ে iPhone app এ offline চালাতে চাও — PyTorch এ train করবে, Core ML এ convert করবে, iOS runtime execute করবে।

### TensorFlow Lite — Android / Edge

Android phone আর embedded devices এর জন্য। Model কে compress করে ছোট করে দেয়।

### MediaPipe — Real-time ML

Google এর runtime। Face detection, hand tracking, pose estimation — এসব real-time ML task এ ব্যবহার হয়। Camera feed এর সাথে কাজ করে।

### NVIDIA TensorRT — Maximum GPU Performance

NVIDIA GPU তে সর্বোচ্চ performance পেতে চাইলে। Model কে NVIDIA GPU এর জন্য specifically optimize করে। Inference speed অনেক বাড়ে।

---

## একটা Runtime এ Train করা Model কি অন্যটায় চলে?

সরাসরি না। PyTorch model TensorFlow এ directly চলবে না। ঠিক যেমন `.py` file Node.js এ চলে না।

কিন্তু convert করা যায়:

```
PyTorch → ONNX → TensorRT (NVIDIA GPU)
PyTorch → Core ML (Apple devices)
PyTorch → TensorFlow Lite (Android)
PyTorch → GGUF (llama.cpp / CPU)
```

PyTorch হলো common starting point — বেশিরভাগ model PyTorch এ train হয়, তারপর target device অনুযায়ী convert হয়।

---

## কোনটা কখন ব্যবহার করবে?

| Scenario | Runtime | কেন |
|----------|---------|-----|
| Development / experiment | PyTorch | সবচেয়ে flexible, biggest ecosystem |
| Local LLM চালানো | llama.cpp / Ollama | CPU তেও চলে, সহজ setup |
| Production LLM serving | vLLM বা TGI | High throughput, batching |
| iPhone/iPad app | Core ML | Neural Engine ব্যবহার করে |
| Android app | TF Lite | Optimized for mobile |
| Maximum GPU speed | TensorRT | NVIDIA specific optimization |
| Cross-platform inference | ONNX Runtime | Universal, CPU efficient |

---

## Developer Analogy — পুরো Stack

```
JavaScript Runtime Stack:
V8 Engine → Node.js → Express → Your API

ML Runtime Stack:
CUDA (GPU driver) → PyTorch → Transformers library → Your model inference
```

আরেকটা:

```
Web Serving:
Code → Webpack (build) → nginx (serve) → Users

ML Serving:
Model → Quantize (optimize) → vLLM (serve) → Users
```

---

## এই Chapter এ কী শিখলে?

Runtime হলো model চালানোর engine — ঠিক যেমন Node.js JavaScript এর runtime। তিনটা major runtime — PyTorch (popular), TensorFlow (enterprise), ONNX (universal)। LLM এর জন্য specialized runtime আছে — llama.cpp (CPU), vLLM (production), TGI (Hugging Face)। Mobile device এ আলাদা runtime লাগে — Core ML, TF Lite, TensorRT। আর PyTorch model convert করে অন্য runtime এ চালানো যায়।

পরের chapter এ আমরা দেখবো Hugging Face — যেখানে এই model গুলো থাকে, download হয়, share হয়। Model এর GitHub।
