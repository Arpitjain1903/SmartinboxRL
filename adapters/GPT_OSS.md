# GPT and Open Source (OSS) Adapters for SmartInboxRL

This adapter provides best practices and settings tuning for deploying the `LLMAgent` with either OpenAI proprietary models (e.g. GPT-4o-mini) or locally deployed Open Source models (e.g. Llama-3, Mistral) via an OpenAI-compatible interface (Ollama, vLLM, Groq).

---

## 1. Running Open-Source Local / Hosted Frameworks

Since `llm_agent.py` uses the standard OpenAI python client, **any OSS proxy or framework that exposes an OpenAI-compatible endpoint works directly.** 

### Ollama (Local OSS)
Run a model: `ollama run llama3.1`
```bash
API_BASE_URL=http://localhost:11434/v1
MODEL_NAME=llama3.1
OPENAI_API_KEY=ollama
```

### Groq (Cloud LLPU for OSS)
Blazing fast execution on Mixtral/Llama for mass-evaluations:
```bash
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.1-8b-instant
OPENAI_API_KEY=gsk_your_groq_api_key_here
```

### vLLM (Custom Inference)
```bash
API_BASE_URL=http://your_vllm_host:8000/v1
MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
OPENAI_API_KEY=empty
```

---

## 2. Tuning for Open Source Models

Smaller (7B-8B parameter) OSS models may struggle slightly with implicit constraint reasoning.

- **Aggressive Formatting Prompts**: Llama 3 and Mistral 7B requires explicit reminders. Add: `You MUST return valid JSON. Do not include markdown codeblocks (\`\`\`json). DO NOT write 'Here is your json'. Just the JSON.`
- **Parsing Weaknesses**: Smaller parameter models frequently misspell priority categories (`cirtical` vs `critical`). The `validate_action()` hook in `environment/action_space.py` handles the worst of it, but enforcing spelling in the system prompt is essential for good evaluation scores.
- **Lower Output Tokens**: Set `max_tokens=64` to `128`. We do not want conversational drift. We are extracting discrete properties.
- **Zero Temperature**: Use `temperature=0.0` or `top_p=0.1`. OSS models at higher temperatures often ignore the structural system prompt.

---

## 3. Tuning for GPT-4o / GPT-4o-mini

OpenAI's official API supports sophisticated features worth leveraging:

### JSON Mode (Response Format)
Instead of relying on prompt parsing, strictly define the JSON response payload required. In `agents/llm_agent.py`:

```python
response = self.client.chat.completions.create(
    model=self.model,
    messages=[
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ],
    temperature=self.temperature,
    max_tokens=256,
    response_format={ "type": "json_object" } # <-- ENSURE STRICT JSON
)
```

Ensure your `_SYSTEM_PROMPT` contains the word "JSON", or the API call will fail when `response_format` is active.

### Context
OpenAI models handle context up to 128k efficiently. The default `history_window=3` (up to 3 emails of trailing history) is perfect. Unlike local OSS models, supplying excess history to GPT-4o does not degrade its instruction-following accuracy.
