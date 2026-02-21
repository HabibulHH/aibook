import ch01 from './chapters/ch01-introduction-tools.md?raw'
import ch02 from './chapters/ch02-data-pipeline.md?raw'
import ch03 from './chapters/ch03-transformer-architecture.md?raw'
import ch04 from './chapters/ch04-model-types-llm-vlm-diffusion.md?raw'
import ch05 from './chapters/ch05-runtime.md?raw'
import ch06 from './chapters/ch06-huggingface-deep-dive.md?raw'
import ch07 from './chapters/ch07-model-prepare-and-store.md?raw'
import ch08 from './chapters/ch08-production-reality.md?raw'
import ch09 from './chapters/ch09-hands-on.md?raw'

export const bookMeta = {
  title: "AI for Developers",
  subtitle: "ডেভেলপারদের জন্য AI এর প্র্যাক্টিক্যাল গাইড",
  author: "PocketSchool",
  description: "Tools দিয়ে শুরু, code দিয়ে শেষ — একটা practical, speaking-tone guide যেখানে AI এর সবকিছু developer এর দৃষ্টিকোণ থেকে।",
  year: 2024,
}

export const chapters = [
  { id: 1, title: "Introduction — Tools দিয়ে শুরু", content: ch01 },
  { id: 2, title: "Data → Model Pipeline", content: ch02 },
  { id: 3, title: "Transformer Architecture", content: ch03 },
  { id: 4, title: "Model Types — LLM, VLM, Diffusion", content: ch04 },
  { id: 5, title: "Runtime — Model চালানোর Engine", content: ch05 },
  { id: 6, title: "Hugging Face Deep Dive", content: ch06 },
  { id: 7, title: "Model Prepare ও Store", content: ch07 },
  { id: 8, title: "Production Reality", content: ch08 },
  { id: 9, title: "Hands-on — Download, Run, API", content: ch09 },
]
