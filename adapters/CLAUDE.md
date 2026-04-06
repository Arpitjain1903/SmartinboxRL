# Claude Adapter for SmartInboxRL

This adapter provides specific guidance for using Claude (e.g., Claude 3.5 Sonnet) as the underlying model for the `LLMAgent` in SmartInboxRL. It helps adapt Claude's natural conversational style into the rigid JSON structure expected by the benchmark.

---

## System Prompt Customization

Claude tends to add conversational filler ("Here is the JSON you requested:") even when explicitly told not to. 

**Recommended Prompt Adjustments:**
When configuring `LLMAgent(model="claude-3-5-sonnet-20240620")`, override the system prompt to enforce strict output formatting using XML constraint cues.

```markdown
You are an intelligent email assistant. Analyze the email and respond with a JSON object.

<formatting_rules>
- You MUST return ONLY valid, parseable JSON.
- DO NOT wrap the json in markdown blocks (```json ... ```).
- DO NOT output any text before the opening `{` or after the closing `}`.
- DO NOT include conversational filler like "Here is the response".
</formatting_rules>

Required JSON keys:
1. "intents": list of valid strings
2. "priority": "low" | "medium" | "high" | "critical"
3. "action": "reply" | "ignore" | "escalate" | "forward"
4. "response": string
```

---

## Token Budgets & Latency

Claude 3.5 Sonnet is highly capable but token volume impacts latency (critical when evaluating 100+ episodes).

- **Input Context:** Moderate (~300-500 tokens per email). The `history_window=3` setting is optimal. Do not increase it past 5, or context cache misses will degrade speed.
- **Max Output Tokens:** Set `max_tokens=256`. The agent only needs to generate a short JSON string. Setting this low prevents runway generation if the model hallucinates formatting.
- **Temperature:** Set `temperature=0.0`. You want deterministic extraction and categorization, not creative variance, to ensure stable benchmark scores.

---

## Parsing Guidance

Because Claude occasionally uses markdown fences despite instructions, ensure `_parse_response` in `llm_agent.py` robustly strips them.

If using the Anthropic API directly (instead of an OpenAI-compatible proxy), use Claude's **Tool Use (Function Calling)** feature instead of raw JSON generation. 

Define the tool schema:
```json
{
  "name": "take_inbox_action",
  "description": "Output the final decision for the current email.",
  "input_schema": {
    "type": "object",
    "properties": {
      "intents": { "type": "array", "items": { "type": "string" } },
      "priority": { "type": "string", "enum": ["low", "medium", "high", "critical"] },
      "action": { "type": "string", "enum": ["reply", "ignore", "escalate", "forward"] },
      "response": { "type": "string" }
    },
    "required": ["intents", "priority", "action", "response"]
  }
}
```
Force the model to use `tool_choice: {"type": "tool", "name": "take_inbox_action"}`. This guarantees perfect JSON extraction without regex parsing fallbacks.
