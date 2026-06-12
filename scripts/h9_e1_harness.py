"""
h9_e1_harness.py — 20-case tool-call harness for H9-E1 Tier 1 baseline (issue #35).

Runtime-agnostic: cases + grader are pure data/functions; thin adapters drive either
the vllm-mlx OpenAI-compatible HTTP endpoint (gated `served` tier) or in-process mlx_lm
(`engine` tier). Grading requires SEMANTIC correctness (key arg values via per-case
validators), not just structural validity — see plan.md Phase B.

Usage:
    uv run python scripts/h9_e1_harness.py --runtime auto    [--port 8123]
    uv run python scripts/h9_e1_harness.py --runtime served  --port 8123
    uv run python scripts/h9_e1_harness.py --runtime engine
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

MODEL = "mlx-community/Qwen3-30B-A3B-4bit"

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-tool JSON — the committed gated API shape)
# ---------------------------------------------------------------------------


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


EMAIL_TRIAGE = _tool(
    "label_email",
    "Apply a triage label and optional route to an incoming email.",
    {
        "label": {
            "type": "string",
            "enum": ["urgent", "follow_up", "newsletter", "spam", "fyi"],
            "description": "Triage label for the email.",
        },
        "archive": {"type": "boolean", "description": "Whether to archive after labeling."},
    },
    ["label"],
)

CALENDAR_LOOKUP = _tool(
    "lookup_calendar",
    "Look up calendar events for a given date (ISO YYYY-MM-DD).",
    {
        "date": {"type": "string", "description": "Date in ISO format YYYY-MM-DD."},
        "attendee": {"type": "string", "description": "Optional attendee email to filter by."},
    },
    ["date"],
)

SLACK_POST = _tool(
    "post_slack",
    "Post a message to a Slack channel.",
    {
        "channel": {"type": "string", "description": "Channel name, including leading #."},
        "text": {"type": "string", "description": "Message body."},
    },
    ["channel", "text"],
)

SEND_EMAIL = _tool(
    "send_email",
    "Send an email to a recipient.",
    {
        "to": {"type": "string", "description": "Recipient email address."},
        "subject": {"type": "string", "description": "Subject line."},
        "body": {"type": "string", "description": "Email body."},
    },
    ["to", "subject", "body"],
)

CREATE_EVENT = _tool(
    "create_event",
    "Create a calendar event.",
    {
        "title": {"type": "string", "description": "Event title."},
        "date": {"type": "string", "description": "Date in ISO format YYYY-MM-DD."},
        "start_time": {"type": "string", "description": "Start time HH:MM 24h."},
    },
    ["title", "date", "start_time"],
)

# Catalog every case can advertise (we pass the relevant subset per case to keep prompts realistic).
ALL_TOOLS = [EMAIL_TRIAGE, CALENDAR_LOOKUP, SLACK_POST, SEND_EMAIL, CREATE_EVENT]


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------


@dataclass
class Case:
    id: str
    category: str
    user_msg: str
    tools: list[dict]
    expected_tool: Optional[str]  # None == no-tool case (correct behavior is a text answer)
    # validators(args_dict) -> bool: checks key argument VALUES (semantic correctness).
    validators: Optional[Callable[[dict], bool]] = None
    system: str = "You are an ops assistant. Use a tool when the task requires one."


def _lc(s: Any) -> str:
    return str(s).strip().lower()


# ---------------------------------------------------------------------------
# 20 cases
# ---------------------------------------------------------------------------

CASES: list[Case] = [
    # --- Email triage / label-route (6) ---
    Case("email-1", "email_triage",
         "This email is from our biggest customer saying production is down and they need a call NOW. Triage it.",
         [EMAIL_TRIAGE], "label_email", lambda a: _lc(a.get("label")) == "urgent"),
    Case("email-2", "email_triage",
         "Got the weekly AWS newsletter digest. File it appropriately.",
         [EMAIL_TRIAGE], "label_email", lambda a: _lc(a.get("label")) == "newsletter"),
    Case("email-3", "email_triage",
         "An email offering me a free cruise and asking for my bank details. Handle it.",
         [EMAIL_TRIAGE], "label_email", lambda a: _lc(a.get("label")) == "spam"),
    Case("email-4", "email_triage",
         "A teammate sent meeting notes 'for your awareness, no action needed'. Label it.",
         [EMAIL_TRIAGE], "label_email", lambda a: _lc(a.get("label")) == "fyi"),
    Case("email-5", "email_triage",
         "A prospect asked a pricing question I should circle back on next week. Label it.",
         [EMAIL_TRIAGE], "label_email", lambda a: _lc(a.get("label")) == "follow_up"),
    Case("email-6", "email_triage",
         "Spam about counterfeit watches — label it spam and archive it.",
         [EMAIL_TRIAGE], "label_email",
         lambda a: _lc(a.get("label")) == "spam" and a.get("archive") is True),

    # --- Calendar lookup (4) ---
    Case("cal-1", "calendar_lookup",
         "What's on my calendar for 2026-06-15?",
         [CALENDAR_LOOKUP], "lookup_calendar", lambda a: a.get("date") == "2026-06-15"),
    Case("cal-2", "calendar_lookup",
         "Show my schedule for July 4th, 2026.",
         [CALENDAR_LOOKUP], "lookup_calendar", lambda a: a.get("date") == "2026-07-04"),
    Case("cal-3", "calendar_lookup",
         "Do I have any meetings with alice@acme.com on 2026-06-20?",
         [CALENDAR_LOOKUP], "lookup_calendar",
         lambda a: a.get("date") == "2026-06-20" and _lc(a.get("attendee")) == "alice@acme.com"),
    Case("cal-4", "calendar_lookup",
         "Check what I have scheduled on the last day of June 2026.",
         [CALENDAR_LOOKUP], "lookup_calendar", lambda a: a.get("date") == "2026-06-30"),

    # --- Slack post (4) ---
    Case("slack-1", "slack_post",
         "Post to the #engineering channel: 'Deploy is green, all checks passed.'",
         [SLACK_POST], "post_slack",
         lambda a: _lc(a.get("channel")).lstrip("#") == "engineering"
                   and "deploy" in _lc(a.get("text")) and "green" in _lc(a.get("text"))),
    Case("slack-2", "slack_post",
         "Let the #sales team know in Slack that the Q3 deck is ready for review.",
         [SLACK_POST], "post_slack",
         lambda a: _lc(a.get("channel")).lstrip("#") == "sales" and "q3" in _lc(a.get("text"))),
    Case("slack-3", "slack_post",
         "Send a Slack message to #general: 'Office closed Friday for the holiday.'",
         [SLACK_POST], "post_slack",
         lambda a: _lc(a.get("channel")).lstrip("#") == "general"
                   and "friday" in _lc(a.get("text"))),
    Case("slack-4", "slack_post",
         "Post in #support that the incident is resolved and the postmortem is scheduled.",
         [SLACK_POST], "post_slack",
         lambda a: _lc(a.get("channel")).lstrip("#") == "support"
                   and ("resolved" in _lc(a.get("text")) or "postmortem" in _lc(a.get("text")))),

    # --- Short summarization: NO-TOOL cases (3). Correct behavior = answer in text, no tool call ---
    Case("sum-1", "summarization_no_tool",
         "Summarize in one sentence: 'The team shipped the new onboarding flow, fixed three "
         "P1 bugs, and started planning the Q3 roadmap.' Just give me the summary.",
         [SLACK_POST, EMAIL_TRIAGE], None, None),
    Case("sum-2", "summarization_no_tool",
         "Give me a one-line TL;DR of this note: 'Customer churn dropped 2% after the pricing "
         "change; support tickets up 10%.' Reply directly, do not use any tool.",
         [CALENDAR_LOOKUP, SLACK_POST], None, None),
    Case("sum-3", "summarization_no_tool",
         "In a sentence, what is the main point: 'Latency improved 30% after the cache rewrite, "
         "but memory use rose.' Answer in plain text.",
         [SEND_EMAIL, CALENDAR_LOOKUP], None, None),

    # --- Mixed / multi-arg (3) ---
    Case("mixed-1", "mixed_multiarg",
         "Email bob@acme.com with subject 'Renewal' telling him the contract renews on July 1.",
         [SEND_EMAIL, SLACK_POST], "send_email",
         lambda a: _lc(a.get("to")) == "bob@acme.com"
                   and "renewal" in _lc(a.get("subject"))
                   and len(_lc(a.get("body"))) > 0),
    Case("mixed-2", "mixed_multiarg",
         "Create a calendar event titled 'Board Review' on 2026-06-18 starting at 14:00.",
         [CREATE_EVENT, CALENDAR_LOOKUP], "create_event",
         lambda a: "board review" in _lc(a.get("title"))
                   and a.get("date") == "2026-06-18"
                   and str(a.get("start_time")).strip() in ("14:00", "2:00 PM", "2:00pm")),
    Case("mixed-3", "mixed_multiarg",
         "Set up a 'Sprint Planning' meeting for 2026-06-22 at 09:30.",
         [CREATE_EVENT, SEND_EMAIL], "create_event",
         lambda a: "sprint planning" in _lc(a.get("title"))
                   and a.get("date") == "2026-06-22"
                   and str(a.get("start_time")).strip() in ("09:30", "9:30", "9:30 AM", "9:30am")),
]


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    case_id: str
    category: str
    structural_valid: bool      # parsed a well-formed tool call (or correctly emitted none)
    semantic_pass: bool         # the GATED metric: right tool + right values, or correct no-tool
    called_tool: Optional[str]
    called_args: Optional[dict]
    raw_preview: str
    error: Optional[str] = None
    recovered_by_lenient: bool = False  # malformed but a lenient parser would have rescued it


def grade(case: Case, parsed_call: Optional[dict], raw_text: str,
          structural_valid: bool, recovered: bool = False) -> CaseResult:
    """parsed_call is {'name': str, 'arguments': dict} or None if no tool call emitted."""
    preview = (raw_text or "")[:160]

    if case.expected_tool is None:
        # No-tool case: pass iff the model did NOT emit a tool call.
        no_call = parsed_call is None
        return CaseResult(case.id, case.category, structural_valid=no_call,
                          semantic_pass=no_call, called_tool=parsed_call and parsed_call.get("name"),
                          called_args=parsed_call and parsed_call.get("arguments"),
                          raw_preview=preview, recovered_by_lenient=recovered)

    # Tool-call case.
    if parsed_call is None:
        return CaseResult(case.id, case.category, structural_valid=False, semantic_pass=False,
                          called_tool=None, called_args=None, raw_preview=preview,
                          error="no tool call emitted", recovered_by_lenient=recovered)

    name = parsed_call.get("name")
    args = parsed_call.get("arguments") or {}
    if not isinstance(args, dict):
        return CaseResult(case.id, case.category, structural_valid=False, semantic_pass=False,
                          called_tool=name, called_args=None, raw_preview=preview,
                          error="arguments not an object", recovered_by_lenient=recovered)

    name_ok = name == case.expected_tool
    try:
        sem_ok = bool(name_ok and (case.validators is None or case.validators(args)))
    except Exception as e:  # validator should never crash on a real dict, but be safe
        sem_ok = False
        return CaseResult(case.id, case.category, structural_valid=structural_valid,
                          semantic_pass=False, called_tool=name, called_args=args,
                          raw_preview=preview, error=f"validator error: {e}",
                          recovered_by_lenient=recovered)

    return CaseResult(case.id, case.category, structural_valid=structural_valid,
                      semantic_pass=sem_ok, called_tool=name, called_args=args,
                      raw_preview=preview, recovered_by_lenient=recovered)


def summarize(results: list[CaseResult]) -> dict:
    n = len(results)
    sem = sum(r.semantic_pass for r in results)
    struct = sum(r.structural_valid for r in results)
    by_cat: dict[str, dict] = {}
    for r in results:
        c = by_cat.setdefault(r.category, {"n": 0, "semantic_pass": 0})
        c["n"] += 1
        c["semantic_pass"] += int(r.semantic_pass)
    return {
        "n_cases": n,
        "semantic_pass": sem,
        "semantic_pass_rate": round(sem / n, 4) if n else 0.0,
        "structural_valid": struct,
        "structural_valid_rate": round(struct / n, 4) if n else 0.0,
        "n_recovered_by_lenient": sum(r.recovered_by_lenient for r in results),
        "gate_pass": (sem / n) >= 0.90 if n else False,
        "by_category": {
            k: {**v, "rate": round(v["semantic_pass"] / v["n"], 4)} for k, v in by_cat.items()
        },
    }


# ---------------------------------------------------------------------------
# Tool-call parsing (normalize both runtimes to {name, arguments})
# ---------------------------------------------------------------------------


def parse_openai_tool_call(message: dict) -> tuple[Optional[dict], bool, bool]:
    """Parse an OpenAI chat-completion message into ({name, arguments}|None, structural_valid, recovered).

    Handles the standard `tool_calls` field; falls back to a lenient JSON-in-content
    scan (Rapid-MLX-style recovery) and reports whether that lenient path was needed.
    """
    tool_calls = message.get("tool_calls")
    if tool_calls:
        fn = tool_calls[0].get("function", {})
        name = fn.get("name")
        raw_args = fn.get("arguments")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            return {"name": name, "arguments": args}, True, False
        except (json.JSONDecodeError, TypeError):
            # Structurally a tool call but arguments not valid JSON.
            return {"name": name, "arguments": {}}, False, False

    # No structured tool_calls — attempt lenient recovery from content.
    content = message.get("content") or ""
    recovered = _lenient_recover(content)
    if recovered is not None:
        return recovered, False, True
    return None, True, False  # no call at all is "structurally fine" (correct for no-tool cases)


def _lenient_recover(content: str) -> Optional[dict]:
    """Best-effort: pull a {"name":..., "arguments":...} JSON object out of free text."""
    if not content or "{" not in content:
        return None
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        return None
    blob = content[start : end + 1]
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and "name" in obj:
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        return {"name": obj["name"], "arguments": args if isinstance(args, dict) else {}}
    return None


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def run_case_served(case: Case, base_url: str) -> tuple[Optional[dict], str, bool, bool]:
    """Drive a case via the vllm-mlx OpenAI-compatible endpoint."""
    import requests

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": case.system},
            {"role": "user", "content": case.user_msg},
        ],
        "tools": case.tools,
        "tool_choice": "auto",
        "temperature": 0.0,
        "max_tokens": 256,
    }
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    message = data["choices"][0]["message"]
    parsed, structural, recovered = parse_openai_tool_call(message)
    raw = json.dumps(message)[:400]
    return parsed, raw, structural, recovered


def run_case_engine(case: Case, model, tokenizer) -> tuple[Optional[dict], str, bool, bool]:
    """Drive a case in-process via mlx_lm + chat template tool support."""
    from mlx_lm import generate

    messages = [
        {"role": "system", "content": case.system},
        {"role": "user", "content": case.user_msg},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tools=case.tools, add_generation_prompt=True, tokenize=False
    )
    text = generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)
    parsed, structural, recovered = _parse_engine_output(text, tokenizer)
    return parsed, text[:400], structural, recovered


def _parse_engine_output(text: str, tokenizer) -> tuple[Optional[dict], bool, bool]:
    """Parse mlx_lm raw generation for a tool call.

    Qwen3's template emits tool calls as <tool_call>{"name":...,"arguments":...}</tool_call>.
    Try that first (structural), then fall back to lenient JSON recovery.
    """
    if "<tool_call>" in text:
        start = text.find("<tool_call>") + len("<tool_call>")
        end = text.find("</tool_call>", start)
        blob = text[start:end] if end > start else text[start:]
        try:
            obj = json.loads(blob.strip())
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            return {"name": obj.get("name"), "arguments": args if isinstance(args, dict) else {}}, True, False
        except (json.JSONDecodeError, TypeError):
            pass  # fall through to lenient
    recovered = _lenient_recover(text)
    if recovered is not None:
        return recovered, False, True
    return None, True, False


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def run_served(base_url: str) -> list[CaseResult]:
    results = []
    for case in CASES:
        try:
            parsed, raw, structural, recovered = run_case_served(case, base_url)
            results.append(grade(case, parsed, raw, structural, recovered))
        except Exception as e:
            results.append(CaseResult(case.id, case.category, False, False, None, None, "", str(e)))
        print(f"  [{results[-1].case_id}] semantic_pass={results[-1].semantic_pass} "
              f"tool={results[-1].called_tool}")
    return results


def run_engine() -> list[CaseResult]:
    from mlx_lm import load

    print(f"  Loading {MODEL} in-process...")
    model, tokenizer = load(MODEL)
    results = []
    for case in CASES:
        try:
            parsed, raw, structural, recovered = run_case_engine(case, model, tokenizer)
            results.append(grade(case, parsed, raw, structural, recovered))
        except Exception as e:
            results.append(CaseResult(case.id, case.category, False, False, None, None, "", str(e)))
        print(f"  [{results[-1].case_id}] semantic_pass={results[-1].semantic_pass} "
              f"tool={results[-1].called_tool}")
    return results


def server_alive(base_url: str) -> bool:
    import requests

    try:
        r = requests.get(f"{base_url}/v1/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", choices=["auto", "served", "engine"], default="auto")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--out", default=None, help="Optional path to dump full per-case JSON.")
    args = ap.parse_args()

    base_url = f"http://localhost:{args.port}"
    runtime = args.runtime
    if runtime == "auto":
        runtime = "served" if server_alive(base_url) else "engine"

    if runtime == "served" and not server_alive(base_url):
        print(f"  vllm-mlx server not reachable at {base_url}; cannot run served tier.")
        return

    gate_tier = "served" if runtime == "served" else "engine"
    print(f"Running harness: runtime={runtime} gate_tier={gate_tier}")
    results = run_served(base_url) if runtime == "served" else run_engine()
    summary = summarize(results)
    summary["runtime"] = runtime
    summary["gate_tier"] = gate_tier

    print("\n=== Tool-call harness summary ===")
    print(json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {"summary": summary, "cases": [r.__dict__ for r in results]}, f, indent=2
            )
        print(f"  Wrote per-case results to {args.out}")

    return summary, results


if __name__ == "__main__":
    main()
