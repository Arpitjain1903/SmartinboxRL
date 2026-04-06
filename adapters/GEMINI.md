# Gemini Adapter for SmartInboxRL

This adapter provides specific guidance for using Google's Gemini models (e.g., Gemini 1.5 Pro, 1.5 Flash) as the underlying model for the `LLMAgent` in SmartInboxRL. It helps adapt Gemini's characteristics into the rigid JSON structure expected by the benchmark.

---

## 1. Context Optimization & Generation Speeds

Gemini natively supports exceptionally large context windows (up to 2M tokens). However, for SmartInboxRL:

- **Do NOT load the whole episode upfront.** A large context drastically increases Time-To-First-Token (TTFT) and evaluation latency.
- Instead, maintain the `history_window=3` (or up to 5) limit in the `inbox_env.py`. This keeps inference latency under 1 second per step.
- **Flash vs. Pro:** Use **Gemini 1.5 Flash** for default baseline evaluations. It is faster, cheaper, and more than capable of handling the straightforward heuristic parsing required by the environment. Reserve **Gemini 1.5 Pro** only if testing the `Hard` tier emails (`data/tasks/hard_tasks.json`) specifically for nuanced intent understanding.

---

## 2. System Prompt Customization for JSON Output

Gemini models tend to excel at instruction following, especially when response formats are declared definitively in the system instructions.

**Recommended Prompt Adjustments:**
```markdown
You are an intelligent email assistant. Analyze the email and respond with a JSON object.

<json_formatting>
Return a clean, valid JSON object ONLY. No markdown (```json), no explanations.
Your JSON must strictly match these exact keys and values:

1. "intents": list of valid strings
2. "priority": "low" | "medium" | "high" | "critical"
3. "action": "reply" | "ignore" | "escalate" | "forward"
4. "response": string
</json_formatting>
```

---

## 3. Using `response_mime_type` (Recommended)

When hitting the Google Gemini API directly (via `google/generative-ai`), do not rely on prompt-engineering alone to ensure JSON parsing in `llm_agent.py`. Instead, natively enforce JSON mode via the generation config.

Update your API call to include:
```python
generation_config = {
    "temperature": 0.0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 256,
    "response_mime_type": "application/json",
}

response = chat_session.send_message(prompt, generation_config=generation_config)
```

Enabling `response_mime_type=application/json` guarantees that the text returned by Gemini is syntactically valid JSON, eliminating the need for complex regex extraction or cleanup.

---

## 4. Grounding

Disable any automated search or Google Grounding functions. For SmartInboxRL, the model must ONLY rely on the immediate text provided in the `body` and `subject`. Grounding leads to hallucinations or off-policy replies (e.g., pulling real-world knowledge instead of sticking strictly to the dummy tasks).
