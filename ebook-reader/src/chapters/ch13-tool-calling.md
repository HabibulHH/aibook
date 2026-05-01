# Module 4: Tool Calling — Agentic Loop Code-এ কীভাবে চলে

> "Module 2-এ tool কী বুঝেছো। Module 3-এ agent কী বুঝেছো। এই module-এ দেখবে — code-এ আসলে দুইটা কীভাবে join হয়।"

> 🎮 **Interactive simulation:** [Whatsmonk Agent Loop দেখো](/simulations/tool-calling.html) — step-by-step loop চালিয়ে দেখতে পারবে। Whatsmonk CRM tools (get_customer_info, send_whatsapp_message, get_recent_leads) দিয়ে agentic loop visualize করা।

---

## আগের দুই Module Recap

আগের module গুলোতে তুমি দুইটা concept আলাদা আলাদা শিখেছো।

**Module 2 (Tools):**
```
Tool = একটা function যেটা LLM call করার decision নেয়
       কিন্তু execute তোমার code করে
```

**Module 3 (Agents):**
```
Agent = LLM + Tools + Loop
        Goal দাও, সে loop চালিয়ে কাজ করে
```

এখন প্রশ্ন — এই দুইটা **code-এ আসলে কীভাবে meet করে?**

Loop টা ঠিক কোথা থেকে শুরু হয়? Tool call response-এ কী আসে? কে message history maintain করে?

এই module-এ একটা real `weather_agent.py` example দিয়ে পুরো flow টা ছিঁড়ে ছিঁড়ে দেখবো।

---

## Tool Calling আর Agent — Relationship টা

আগে relationship টা পরিষ্কার করি।

```
┌──────────────────────────────────────────────┐
│              Agent                            │
│                                               │
│   ┌──────────────────────────────────────┐  │
│   │         Agentic Loop                  │  │
│   │                                       │  │
│   │   ┌─────────────────────────────┐   │  │
│   │   │     Tool Calling             │   │  │
│   │   │  (একবারের request/response)  │   │  │
│   │   └─────────────────────────────┘   │  │
│   │                                       │  │
│   │   একই tool calling বারবার চলে        │  │
│   └──────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

**Tool Calling** = একটা single API call-এ LLM tool use করতে চাওয়া।
**Agentic Loop** = এই tool calling বারবার চালানো — যতক্ষণ না goal achieve হয়।
**Agent** = পুরো system — loop + tools + state।

মানে — tool calling **agent-এর engine।** Loop ছাড়া tool calling = chatbot। Loop সহ tool calling = agent।

---

## Real Example — Weather Agent

আমরা একটা actual Python file দেখবো — `weather_agent.py`। ছোট, কিন্তু production-grade pattern।

```python
import json
import sys
import httpx
from anthropic import Anthropic

MODEL = "claude-opus-4-7"

tools = [
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for a given location. "
            "Returns temperature (°C), wind speed (km/h), and a short condition summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'Dhaka', 'San Francisco', 'Tokyo'.",
                }
            },
            "required": ["location"],
        },
    }
]
```

এই tool definition-টা module 2 থেকে চেনা। তিনটা field — `name`, `description`, `input_schema`। কিছু নতুন না।

---

## Step 1 — Tool Definition কেন এত Important

আগে এই block-টা মনোযোগ দিয়ে পড়ো।

```python
"input_schema": {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "City name, e.g. 'Dhaka', 'San Francisco', 'Tokyo'.",
        }
    },
    "required": ["location"],
}
```

এটা JSON Schema। LLM-এর জন্য এটা একটা **contract।**

**Schema-র তিনটা কাজ:**

```
1. LLM-কে বলে: "এই tool কী কী parameter নেয়"
2. LLM-এর output dictate  করে: তোমার শেষ output এই shape-এই হবে
3. Description দিয়ে LLM-কে guide করে: কী value পাঠাতে হবে
```

Schema না দিলে কী হবে? LLM **guess করবে।** কখনো `{"city": "Dhaka"}` পাঠাবে, কখনো `{"place": "Dhaka"}`, কখনো `{"query": "weather Dhaka"}`। তোমার code break হয়ে যাবে।

Schema **predictability** নিশ্চিত করে।

**Description-এর role:**

দুই জায়গায় description লাগে — দুইটার কাজ আলাদা।

| Field | LLM কখন পড়ে | কাজ |
|---|---|---|
| Top-level `description` | Tool select করার সময় | "এই tool টা কখন call করবো?" |
| Property `description` | Argument generate করার সময় | "এই field-এ কী value দিবো?" |

`"City name, e.g. 'Dhaka', 'San Francisco', 'Tokyo'"` — এই hint-টা LLM-কে বলে "country না, **city** পাঠাও।" তুমি যদি লিখতে `"Location identifier"`, তাহলে LLM হয়তো `"BD"` পাঠাতো — তোমার geocoding API সেটা হ্যান্ডেল করতে পারতো না।

---

## Step 2 — Local Tool Function

```python
def get_weather(location: str) -> dict:
    geo = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    ).json()

    results = geo.get("results") or []
    if not results:
        return {"error": f"Could not find location '{location}'."}

    place = results[0]
    lat, lon = place["latitude"], place["longitude"]

    wx = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current_weather": True},
        timeout=10,
    ).json()

    cur = wx["current_weather"]
    return {
        "location": f"{place['name']}, {place.get('country', '')}",
        "temperature_c": cur["temperature"],
        "wind_kph": cur["windspeed"],
        "condition": cur["weathercode"],
    }
```

এটা একটা সাধারণ Python function। **LLM সম্পর্কে কিছুই জানে না।**

City name নেয় → geocoding API call করে → forecast API call করে → dict return করে।

মানে — তোমার tool implementation **LLM-agnostic।** তুমি এই function কে CLI থেকেও call করতে পারো, FastAPI থেকেও call করতে পারো, LLM থেকেও call করতে পারো। Same function।

**Important point:** Tool function কোনো magic না। সাধারণ code। Magic শুধু **decision-making** layer-এ।

---

## Step 3 — Agentic Loop টা

এখন আসল part। `run_agent` function।

```python
def run_agent(user_message: str) -> str:
    client = Anthropic()
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "get_weather":
                    result = get_weather(block.input["location"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            continue

        return "".join(b.text for b in response.content if b.type == "text")
```

ছোট। ৩০ লাইনও না। কিন্তু এটাই **agent-এর core।**

ছিঁড়ে ছিঁড়ে দেখি।

---

## Anatomy 1 — Conversation State

```python
messages = [{"role": "user", "content": user_message}]
```

`messages` একটা list। **এটাই agent-এর memory।**

LLM stateless। তুমি প্রতিবার পুরো history পাঠাও, সে শেষ message-এর reply করে। তাই `messages` list-এ history accumulate হতে থাকে।

প্রতি iteration-এ এই list **grow করে।** Loop শেষে list-এ থাকবে:

```
[
  user message,
  assistant tool call 1,
  user (tool result 1),
  assistant tool call 2,
  user (tool result 2),
  ...
  assistant final answer
]
```

---

## Anatomy 2 — The API Call

```python
response = client.messages.create(
    model=MODEL,
    max_tokens=1024,
    tools=tools,
    messages=messages,
)
```

প্রতিবার যা পাঠাচ্ছো:
- **model** — কোন model use করবে
- **tools** — available tools এর list (LLM এটা পড়ে decide করে কোনটা call করবে)
- **messages** — পুরো conversation history
- **max_tokens** — output limit

**Important:** `tools` list **প্রতিবার পাঠাতে হয়।** তুমি ভেবো না LLM আগেরবার দেখেছে — সে stateless, প্রতি call-এ আবার দেখাও।

---

## Anatomy 3 — Response Shape

LLM যা return করে সেটা একটা `Message` object। দুই rokomer response আসতে পারে।

**Case 1 — LLM tool call করতে চায়:**

```python
Message(
    id="msg_01XyZ...",
    role="assistant",
    stop_reason="tool_use",
    content=[
        ToolUseBlock(
            type="tool_use",
            id="toolu_01ABcDeFgHiJkLmNoPqRsTu",
            name="get_weather",
            input={"location": "Dhaka"}
        ),
    ],
)
```

**Case 2 — LLM final answer দিচ্ছে:**

```python
Message(
    id="msg_01AbC...",
    role="assistant",
    stop_reason="end_turn",
    content=[
        TextBlock(
            type="text",
            text="Dhaka-এর current temperature 31°C..."
        ),
    ],
)
```

**Key observations:**

```
1. response.content একটা LIST
   → একই response-এ multiple block থাকতে পারে
   → text + tool_use mix হতে পারে
   → একাধিক tool_use থাকতে পারে (parallel calls)

2. stop_reason বলে কী হলো
   → "tool_use" = LLM tool চায়, তোমার কাজ আছে
   → "end_turn" = LLM শেষ, final answer দিয়েছে
   → "max_tokens" = limit hit করেছে, output incomplete
```

---

## Anatomy 4 — `ToolUseBlock` কোথা থেকে আসে

```python
ToolUseBlock(
    type="tool_use",
    id="toolu_01ABcDeFgHiJkLmNoPqRsTu",
    name="get_weather",
    input={"location": "Dhaka"}
)
```

এটা কে generate করে?

**LLM নিজে।** Token by token। Just like text generation।

LLM তোমার `input_schema` পড়ে। Decide করে — "এই user message-এ tool call দরকার, schema মতে input বানাই।" তারপর JSON shape-এ output করে।

| Field | কে decide করে |
|---|---|
| `type` | LLM (tool call করার সিদ্ধান্ত) |
| `id` | API generate করে (unique identifier) |
| `name` | LLM (available tools list থেকে) |
| `input` | LLM (input_schema match করে) |

**মানে — তোমার code কিচ্ছু "fill in" করে না।** Tool calling মানে LLM-কে structured output generate করানো, যেটা তোমার code execute করে।

---

## Anatomy 5 — Loop Logic

```python
if response.stop_reason == "tool_use":
    # Tool execute করো
    tool_results = []
    for block in response.content:
        if block.type == "tool_use" and block.name == "get_weather":
            result = get_weather(block.input["location"])
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })

    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": tool_results})
    continue

return "".join(b.text for b in response.content if b.type == "text")
```

পরিষ্কার করে দেখি:

**1. Check — LLM কি tool চায়?**
```python
if response.stop_reason == "tool_use":
```
`stop_reason` যদি `"tool_use"` না হয়, মানে LLM tool চায় না — final answer দিয়েছে। সরাসরি return।

**2. Iterate — সব tool call execute করো**
```python
for block in response.content:
    if block.type == "tool_use" and block.name == "get_weather":
        result = get_weather(block.input["location"])
```
`response.content` একটা list। Multiple tool call থাকতে পারে। সব iterate করে execute করো।

**3. Result wrap করো — `tool_use_id` দিয়ে link**
```python
tool_results.append({
    "type": "tool_result",
    "tool_use_id": block.id,        # ← এটা important
    "content": json.dumps(result),
})
```
`tool_use_id` দিয়ে LLM বুঝবে "কোন call-এর reply এটা।" Multiple parallel calls হলে এই id-ই matching করে।

**4. History update করো**
```python
messages.append({"role": "assistant", "content": response.content})
messages.append({"role": "user", "content": tool_results})
```

দুইটা message append হলো:
- **Assistant turn** — LLM-এর tool request (verbatim, যেমন এসেছিল)
- **User turn** — tool result (API convention — tool result কে user message-এ পাঠাতে হয়)

**5. `continue` — loop iterate করো**
পরের iteration-এ updated `messages` যাবে। LLM tool result দেখবে, তারপর হয় আরেকটা tool call করবে, না হয় final answer দিবে।

---

## Visual — পুরো Flow

```
User: "What's the weather in Dhaka?"
        ↓
┌───────────────────────────────────────────┐
│ Iteration 1                                │
│                                            │
│ messages = [user msg]                     │
│ → API call                                │
│ ← response: stop_reason="tool_use"        │
│   content: [ToolUseBlock(location="Dhaka")]│
│                                            │
│ get_weather("Dhaka") execute              │
│ → {temp: 31, condition: "Partly cloudy"}  │
│                                            │
│ messages.append(assistant tool call)      │
│ messages.append(user tool result)         │
│ continue ↓                                │
└───────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────┐
│ Iteration 2                                │
│                                            │
│ messages = [user, assistant, user_result] │
│ → API call                                │
│ ← response: stop_reason="end_turn"        │
│   content: [TextBlock("Dhaka-এ 31°C...")] │
│                                            │
│ stop_reason != "tool_use" → return text   │
└───────────────────────────────────────────┘
        ↓
"Dhaka-এর current temperature 31°C..."
```

**Pattern:** Request → Tool Call → Result → Request → Final Answer। যতবার দরকার, ততবার iterate।

---

## Edge Cases

**Case A — Tool দরকার নাই**

```python
User: "Hello, how are you?"
```

LLM weather tool call করবে না। সরাসরি response:
```python
Message(
    stop_reason="end_turn",
    content=[TextBlock("I'm doing well, thanks!")]
)
```

`if response.stop_reason == "tool_use"` false → loop break, return text।

**Tools available থাকলেও LLM forced না।** সে নিজে decide করে।

**Case B — Multiple Parallel Tool Calls**

```python
User: "Compare weather in Dhaka and Tokyo"
```

LLM একই response-এ দুইটা tool call করতে পারে:
```python
content=[
    ToolUseBlock(id="toolu_01A", input={"location": "Dhaka"}),
    ToolUseBlock(id="toolu_01B", input={"location": "Tokyo"}),
]
```

Loop already handle করে — `for block in response.content` দুইবার চলবে, দুইটা result accumulate হবে। এক user message-এ দুইটা `tool_result` যাবে।

**Case C — Multi-Turn Tool Calling**

```python
User: "Compare today's Dhaka weather with yesterday's"
```

এখানে LLM হয়তো:
1. প্রথমে today's weather call করবে
2. Result দেখে decide করবে "yesterday-র জন্য আরেকটা tool লাগবে"
3. দ্বিতীয় call করবে
4. দুইটা result compare করে answer দিবে

Loop এই sequential pattern handle করে। যতক্ষণ `stop_reason == "tool_use"`, ততক্ষণ continue।

**Case D — Unknown Tool Name**

```python
if block.type == "tool_use" and block.name == "get_weather":
```

`get_weather` ছাড়া অন্য কোনো tool name এলে কী হবে? Code silently skip করবে। Result-এ সেই tool-এর entry থাকবে না। LLM পরের iteration-এ confused হতে পারে।

Production code-এ এই case explicitly handle করতে হয় — error message return করো বা proper logging করো।

---

## Tool Calling Loop — Architectural Summary

```
┌────────────────────────────────────────────┐
│  while True:                                │
│      response = LLM.create(messages)       │
│                                             │
│      if response wants tool:                │
│          execute tool                       │
│          append to messages                 │
│          continue                           │
│      else:                                  │
│          return final text                  │
│          break                              │
└────────────────────────────────────────────┘
```

**এই pattern টা যেকোনো agent-এ same।** Coding agent, browser agent, support agent — sob er core এই loop।

পার্থক্য শুধু:
- কী কী tool আছে
- Tool কী কী করে
- Loop কখন থামে (max iterations? user approval?)

---

## কেন এই pattern Powerful

```
1. Composability:
   LLM নিজে compose করে — কোন tool কখন use করবে।
   তুমি sequence hard-code করো না।

2. Adaptability:
   Tool result দেখে LLM next step decide করে।
   Result fail করলে fallback try করতে পারে।

3. Extensibility:
   নতুন tool add করলে code change লাগে না।
   শুধু tools list-এ entry আর function add।
```

এটাই traditional API integration আর agent-এর মধ্যে gap। API তে তুমি সব sequence লেখো। Agent-এ LLM compose করে।

---

## Production-এ যা লাগবে

এই simple loop **demo-র জন্য fine।** Production-এ আরো লাগবে:

```
1. Max iteration limit:
   while iterations < 10:
   → infinite loop থেকে বাঁচতে

2. Error handling:
   Tool call failed হলে কী?
   API timeout হলে কী?

3. Logging/observability:
   কোন tool কতবার call হলো
   কত time লাগলো
   কী input, কী output

4. Cost tracking:
   প্রতি iteration token খরচ করে
   Cap রাখা দরকার

5. Tool approval (sometimes):
   Destructive tool (delete, send_email)
   User approval নেওয়া

6. Streaming:
   Long response stream করা UX-এর জন্য
```

Module 9 (Production)-এ এই গুলোতে details যাবো।

---

## এক লাইনে Module 4

```
Tool Calling   = একটা single LLM call যেখানে structured tool request আসে
Agentic Loop   = এই tool calling বারবার চালানো, history maintain করে
Agent          = পুরো system — loop + tools + state + error handling

Code-এ pattern একটাই:
while True:
    response = LLM(messages + tools)
    if response wants tool:
        execute, append result, continue
    else:
        return final answer
```

মনে রাখো:
- LLM **decide করে** কোন tool, কী input
- তোমার code **execute করে** actual function
- `messages` list **state maintain করে**
- `stop_reason` **loop control করে**

কিন্তু এই loop যত বড় হবে, `messages` list তত বড় হতে থাকবে। 50 iteration পরে list-এ 100+ messages। Token খরচ blow up করবে। Context window overflow করবে।

এটাই পরের module — **Context আর Memory — State কে কীভাবে manage করে।**
