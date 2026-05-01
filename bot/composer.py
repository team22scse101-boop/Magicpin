"""
LLM-powered message composer — the core engine.
Uses Gemini Flash to compose messages grounded in context.
"""
import json
import re
import os
import time
from typing import Any, Dict, Optional
from urllib import request as urlrequest, error as urlerror

from bot.config import LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT

# NVIDIA API config (OpenAI-compatible)
_NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
_NVIDIA_MODEL = "google/gemma-3n-e4b-it"
_NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Retry config
_MAX_RETRIES = 3
_RETRY_DELAYS = [1, 2, 4]  # seconds


def _call_llm(prompt: str, system: str = "") -> str:
    """Call NVIDIA API (OpenAI-compatible) with retry for rate limits."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = json.dumps({
        "model": _NVIDIA_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "stream": False,
    }).encode("utf-8")

    for attempt in range(_MAX_RETRIES):
        try:
            req = urlrequest.Request(_NVIDIA_URL, data=body, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_NVIDIA_API_KEY}",
            })
            resp = urlrequest.urlopen(req, timeout=LLM_TIMEOUT)
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except urlerror.HTTPError as e:
            if e.code == 429 and attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAYS[attempt])
                continue
            raise
        except Exception:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAYS[attempt])
                continue
            raise


def _extract_json(text: str) -> dict:
    """Extract first JSON object from LLM response."""
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _build_voice_instructions(category: dict) -> str:
    """Build voice/tone instructions from category context."""
    voice = category.get("voice", {})
    tone = voice.get("tone", "professional")
    taboos = voice.get("vocab_taboo", [])
    allowed = voice.get("vocab_allowed", [])[:10]
    taboo_str = ", ".join(f'"{t}"' for t in taboos[:6]) if taboos else "none"
    allowed_str = ", ".join(allowed) if allowed else "general"
    return (
        f"TONE: {tone}. "
        f"ALLOWED VOCABULARY: {allowed_str}. "
        f"NEVER USE THESE WORDS/PHRASES: {taboo_str}. "
    )


def _build_merchant_summary(merchant: dict) -> str:
    """Build a concise merchant summary for the prompt."""
    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    delta = perf.get("delta_7d", {})
    offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    signals = merchant.get("signals", [])
    cust_agg = merchant.get("customer_aggregate", {})
    review_themes = merchant.get("review_themes", [])
    conv_hist = merchant.get("conversation_history", [])
    sub = merchant.get("subscription", {})

    parts = [
        f"Name: {identity.get('name', '?')}",
        f"Owner: {identity.get('owner_first_name', '?')}",
        f"City: {identity.get('city', '?')}, Locality: {identity.get('locality', '?')}",
        f"Languages: {identity.get('languages', ['en'])}",
        f"Subscription: {sub.get('status', '?')} ({sub.get('plan', '?')})",
    ]
    if sub.get("days_remaining"):
        parts.append(f"Days remaining: {sub['days_remaining']}")
    if sub.get("days_since_expiry"):
        parts.append(f"Days since expiry: {sub['days_since_expiry']}")

    parts.append(f"Performance (30d): views={perf.get('views', '?')}, calls={perf.get('calls', '?')}, "
                 f"directions={perf.get('directions', '?')}, CTR={perf.get('ctr', '?')}")
    if delta:
        parts.append(f"7d deltas: views {delta.get('views_pct', 0):+.0%}, calls {delta.get('calls_pct', 0):+.0%}")

    if offers:
        parts.append(f"Active offers: {', '.join(offers)}")
    else:
        parts.append("Active offers: NONE")

    if signals:
        parts.append(f"Signals: {', '.join(signals[:6])}")

    if cust_agg:
        cust_parts = []
        for k, v in cust_agg.items():
            cust_parts.append(f"{k}={v}")
        parts.append(f"Customer aggregate: {', '.join(cust_parts)}")

    if review_themes:
        for rt in review_themes[:3]:
            parts.append(f"Review theme: {rt.get('theme')} ({rt.get('sentiment')}, {rt.get('occurrences_30d', 0)}x)")
            if rt.get("common_quote"):
                parts.append(f'  Quote: "{rt["common_quote"]}"')

    if conv_hist:
        for turn in conv_hist[-3:]:
            parts.append(f"Conv [{turn.get('from')}]: \"{turn.get('body', '')[:120]}\" ({turn.get('engagement', '')})")

    return "\n".join(parts)


def _build_trigger_summary(trigger: dict) -> str:
    """Build a concise trigger summary."""
    parts = [
        f"Trigger kind: {trigger.get('kind', '?')}",
        f"Source: {trigger.get('source', '?')}",
        f"Scope: {trigger.get('scope', '?')}",
        f"Urgency: {trigger.get('urgency', '?')}/5",
        f"Suppression key: {trigger.get('suppression_key', '')}",
    ]
    payload = trigger.get("payload", {})
    if payload:
        parts.append(f"Payload: {json.dumps(payload, ensure_ascii=False)}")
    return "\n".join(parts)


def _build_category_digest(category: dict, trigger: dict) -> str:
    """Build relevant digest items from category."""
    digest = category.get("digest", [])
    if not digest:
        return "No digest items available."

    # Find specific digest item referenced by trigger
    top_item_id = trigger.get("payload", {}).get("top_item_id") or trigger.get("payload", {}).get("digest_item_id")
    parts = []
    for item in digest:
        if top_item_id and item.get("id") == top_item_id:
            parts.insert(0, f"[RELEVANT] {item.get('title', '')} — Source: {item.get('source', '')}. "
                           f"{item.get('summary', '')} "
                           f"(trial_n={item.get('trial_n', 'N/A')}, segment={item.get('patient_segment', 'N/A')})")
        else:
            parts.append(f"- {item.get('title', '')} ({item.get('source', '')})")

    return "\n".join(parts[:5])


def _build_customer_summary(customer: dict) -> str:
    """Build a concise customer summary."""
    if not customer:
        return "No customer context (merchant-facing message)."
    identity = customer.get("identity", {})
    rel = customer.get("relationship", {})
    prefs = customer.get("preferences", {})
    consent = customer.get("consent", {})
    parts = [
        f"Name: {identity.get('name', '?')}",
        f"Language: {identity.get('language_pref', 'en')}",
        f"Age band: {identity.get('age_band', '?')}",
        f"State: {customer.get('state', '?')}",
        f"Visits: {rel.get('visits_total', 0)}, First: {rel.get('first_visit', '?')}, Last: {rel.get('last_visit', '?')}",
        f"Services: {rel.get('services_received', [])}",
        f"LTV: ₹{rel.get('lifetime_value', 0)}",
        f"Preferred slots: {prefs.get('preferred_slots', '?')}",
        f"Channel: {prefs.get('channel', 'whatsapp')}",
        f"Consent scope: {consent.get('scope', [])}",
    ]
    # Extra fields
    if prefs.get("wedding_date"):
        parts.append(f"Wedding date: {prefs['wedding_date']}")
    if prefs.get("training_focus"):
        parts.append(f"Training focus: {prefs['training_focus']}")
    if identity.get("senior_citizen"):
        parts.append("Senior citizen: YES")
    if prefs.get("delivery_address") == "saved":
        parts.append("Delivery address: SAVED")
    if prefs.get("preferred_stylist"):
        parts.append(f"Preferred stylist: {prefs['preferred_stylist']}")
    return "\n".join(parts)


def _get_peer_stats_summary(category: dict) -> str:
    """Summarize peer stats for comparison."""
    ps = category.get("peer_stats", {})
    if not ps:
        return ""
    parts = []
    for k, v in ps.items():
        if k != "scope":
            parts.append(f"{k}={v}")
    return f"Peer benchmarks ({ps.get('scope', 'general')}): {', '.join(parts)}"

def _get_trigger_framing(kind: str, merchant: dict, trigger: dict, customer: Optional[dict]) -> str:
    """Return trigger-specific prompt instructions."""
    owner = merchant.get("identity", {}).get("owner_first_name", "")
    perf = merchant.get("performance", {})
    delta = perf.get("delta_7d", {})
    payload = trigger.get("payload", {})

    framings = {
        "perf_spike": f"ANGLE: Celebrate the spike. Cite exact numbers (views={perf.get('views')}, calls={perf.get('calls')}). Show the 7d delta. Ask if they want to amplify with an offer. CTA: binary_yes_no.",
        "perf_dip": f"ANGLE: Urgent but supportive. Cite the drop ({delta.get('views_pct', 0):+.0%} views). Compare to peer benchmarks. Suggest ONE specific fix (new offer, photo update, or boost). CTA: binary_yes_no.",
        "renewal_due": f"ANGLE: Subscription urgency. Mention days remaining. Cite what they'd lose (visibility, calls). Reference their best metric. CTA: binary_confirm_cancel.",
        "dormant_with_vera": "ANGLE: Re-engagement. Reference their last conversation topic. Ask a single low-effort question to restart. CTA: binary_yes_no.",
        "milestone_reached": f"ANGLE: Celebration + upsell. Cite the exact milestone from payload. Congratulate {owner}. Suggest sharing as a social proof post. CTA: binary_yes_no.",
        "review_theme_emerged": "ANGLE: Cite the exact review theme and quote. If positive, suggest amplifying. If negative, suggest addressing. CTA: open_ended.",
        "competitor_opened": "ANGLE: Competitive urgency. Mention the new competitor. Suggest a defensive action (boost, new offer, profile update). CTA: binary_yes_no.",
        "festival_upcoming": "ANGLE: Seasonal opportunity. Name the specific festival. Suggest a themed offer or campaign. Create FOMO with timing. CTA: binary_yes_no.",
        "recall_due": "ANGLE: Customer care. Reference the specific patient/customer and their last visit. Suggest sending a recall message. CTA: binary_confirm_cancel.",
        "customer_lapsed_soft": "ANGLE: Win-back the customer. Cite visit history and time since last visit. Draft a personalized re-engagement message. CTA: binary_confirm_cancel.",
        "customer_lapsed_hard": "ANGLE: Last-chance win-back. Cite total lifetime value. Suggest a special offer to bring them back. CTA: binary_yes_no.",
        "appointment_tomorrow": "ANGLE: Reminder. Confirm the appointment details. Keep it short and warm. CTA: binary_confirm_cancel.",
        "chronic_refill_due": "ANGLE: Health reminder. Reference the specific medication and refill schedule. Suggest sending a refill reminder. CTA: binary_confirm_cancel.",
        "trial_followup": "ANGLE: Convert trial to paid. Reference the trial details and experience. Ask about conversion. CTA: binary_yes_no.",
        "research_digest": "ANGLE: Share a relevant industry insight from the digest. Connect it to their practice. Suggest an action. CTA: open_ended.",
        "winback_eligible": "ANGLE: Re-engage lapsed merchant. Reference their past performance. Show what they're missing. CTA: binary_yes_no.",
        "supply_alert": "ANGLE: Urgent supply notification. Cite the specific product/supply issue. Suggest immediate action. CTA: binary_yes_no.",
        "curious_ask_due": "ANGLE: Proactive check-in. Ask about a specific aspect of their business. Low-pressure, curiosity-driven. CTA: open_ended.",
        "category_seasonal": "ANGLE: Seasonal trend. Cite the specific seasonal pattern. Suggest capitalizing on it now. CTA: binary_yes_no.",
        "gbp_unverified": "ANGLE: Business profile urgency. Explain visibility loss from unverified profile. Offer to help verify. CTA: binary_yes_no.",
        "cde_opportunity": "ANGLE: Educational opportunity. Reference the specific event/webinar. Explain the benefit. CTA: binary_yes_no.",
    }
    return framings.get(kind, f"ANGLE: Address the {kind} trigger directly. Cite specific numbers. One clear CTA.")


def compose(category: dict, merchant: dict, trigger: dict,
            customer: Optional[dict] = None) -> dict:
    """
    Compose a message from the 4 contexts.
    Returns dict with: body, cta, send_as, suppression_key, rationale, template_name, template_params
    """
    is_customer_facing = customer is not None and trigger.get("scope") == "customer"
    send_as = "merchant_on_behalf" if is_customer_facing else "vera"
    kind = trigger.get("kind", "general")

    system_prompt = f"""You are Vera, magicpin's AI merchant growth assistant. You compose WhatsApp messages for Indian merchants.

ROLE: You compose ONE message based on the context below. Your output must be a JSON object.

CRITICAL RULES:
1. Use ONLY facts from the provided context. NEVER fabricate numbers, dates, names, or sources.
2. Keep the message concise — aim for 3-5 sentences max.
3. Use exactly ONE primary CTA (call-to-action) at the end.
4. Match the merchant's language preference. If languages include "hi", use natural Hindi-English code-mix.
5. Use the owner's first name (not generic "Hi" or "Dear merchant").
6. Reference specific numbers from their performance, offers, or customer data.
7. Explain WHY NOW — connect the message to the specific trigger.
8. No long preambles like "I hope you're doing well".
9. No multiple CTAs. One clear next step.
10. If sending as merchant_on_behalf (customer-facing), use the merchant's name/clinic name, not "Vera".
11. Never re-introduce yourself after the first message.

{_build_voice_instructions(category)}
{_get_peer_stats_summary(category)}
"""

    user_prompt = f"""COMPOSE A MESSAGE for this context:

=== CATEGORY ===
Slug: {category.get('slug', '?')}
Digest items:
{_build_category_digest(category, trigger)}

Seasonal beats: {json.dumps(category.get('seasonal_beats', []), ensure_ascii=False)}
Trend signals: {json.dumps(category.get('trend_signals', [])[:3], ensure_ascii=False)}

=== MERCHANT ===
{_build_merchant_summary(merchant)}

=== TRIGGER ===
{_build_trigger_summary(trigger)}

=== CUSTOMER ===
{_build_customer_summary(customer)}

=== TASK ===
Compose the WhatsApp message. Send as: {send_as}.
Trigger kind: {kind}.

{_get_trigger_framing(kind, merchant, trigger, customer)}

Return ONLY this JSON (no markdown, no extra text):
{{
  "body": "<the WhatsApp message body>",
  "cta": "<one of: binary_yes_no, binary_confirm_cancel, open_ended, multi_choice_slot, none>",
  "rationale": "<1-2 sentences: why this message, what signal drives it, what it should achieve>"
}}"""

    try:
        raw = _call_llm(user_prompt, system_prompt)
        result = _extract_json(raw)

        if not result.get("body"):
            return _fallback_compose(category, merchant, trigger, customer, send_as)

        body = result["body"]
        cta = result.get("cta", "open_ended")
        rationale = result.get("rationale", "Composed from context")

        # Build template params from body
        owner = merchant.get("identity", {}).get("owner_first_name", "")
        name = merchant.get("identity", {}).get("name", "")
        template_params = [owner or name, body[:100], body[100:200] if len(body) > 100 else ""]

        return {
            "body": body,
            "cta": cta,
            "send_as": send_as,
            "suppression_key": trigger.get("suppression_key", f"{kind}:{merchant.get('merchant_id', '')}"),
            "rationale": rationale,
            "template_name": f"vera_{kind}_v1",
            "template_params": template_params,
        }

    except Exception as e:
        return _fallback_compose(category, merchant, trigger, customer, send_as, str(e))


def _fallback_compose(category: dict, merchant: dict, trigger: dict,
                      customer: Optional[dict], send_as: str, error: str = "") -> dict:
    """Deterministic fallback when LLM fails."""
    identity = merchant.get("identity", {})
    owner = identity.get("owner_first_name", identity.get("name", ""))
    kind = trigger.get("kind", "update")
    perf = merchant.get("performance", {})

    if send_as == "merchant_on_behalf" and customer:
        cust_name = customer.get("identity", {}).get("name", "")
        clinic_name = identity.get("name", "")
        body = (f"Hi {cust_name}, {clinic_name} here. "
                f"We have an update for you based on your recent visit. "
                f"Reply YES if you'd like to know more.")
    else:
        body = (f"{owner}, quick update — your profile has {perf.get('views', 0)} views "
                f"and {perf.get('calls', 0)} calls this month. "
                f"Want me to help improve these numbers?")

    return {
        "body": body,
        "cta": "binary_yes_no",
        "send_as": send_as,
        "suppression_key": trigger.get("suppression_key", f"{kind}:{merchant.get('merchant_id', '')}"),
        "rationale": f"Fallback composition (LLM error: {error[:100]})" if error else "Fallback composition",
        "template_name": f"vera_{kind}_fallback_v1",
        "template_params": [owner, str(perf.get("views", 0)), str(perf.get("calls", 0))],
    }


def compose_reply(category: dict, merchant: dict, trigger: dict,
                  conversation_history: list, merchant_message: str,
                  customer: Optional[dict] = None) -> dict:
    """
    Compose a reply to a merchant/customer message within an existing conversation.
    Returns dict with: action, body, cta, rationale
    """
    is_customer_facing = customer is not None
    send_as = "merchant_on_behalf" if is_customer_facing else "vera"

    # Build conversation history string
    conv_str = ""
    for turn in conversation_history[-6:]:
        role = turn.get("from", turn.get("from_role", "?"))
        msg = turn.get("body", turn.get("msg", ""))[:150]
        conv_str += f"[{role}]: {msg}\n"

    system_prompt = f"""You are Vera, magicpin's AI merchant growth assistant, in an active WhatsApp conversation.

RULES:
1. You are replying to the merchant's latest message.
2. Be helpful, concise, and action-oriented.
3. If the merchant says YES or commits, switch to ACTION mode immediately — draft, schedule, send, confirm. DO NOT ask more qualifying questions.
4. If the merchant asks something off-topic (GST, legal, etc.), politely decline and redirect.
5. If the merchant is hostile or says stop, exit gracefully.
6. Use facts from context only. Never fabricate.
7. Match their language style.

{_build_voice_instructions(category)}
"""

    user_prompt = f"""CONVERSATION SO FAR:
{conv_str}
[merchant]: {merchant_message}

=== MERCHANT CONTEXT ===
{_build_merchant_summary(merchant)}

=== ORIGINAL TRIGGER ===
Kind: {trigger.get('kind', '?')}
Payload: {json.dumps(trigger.get('payload', {}), ensure_ascii=False)}

=== TASK ===
Compose Vera's reply. Return ONLY JSON:
{{
  "action": "<send|wait|end>",
  "body": "<reply text, empty if action is wait or end>",
  "cta": "<binary_yes_no|open_ended|none>",
  "rationale": "<why this response>"
}}"""

    try:
        raw = _call_llm(user_prompt, system_prompt)
        result = _extract_json(raw)

        if not result.get("action"):
            return {"action": "send", "body": "Got it, let me work on that for you.",
                    "cta": "none", "rationale": "Fallback acknowledgment"}

        resp = {"action": result["action"], "rationale": result.get("rationale", "")}
        if result["action"] == "send":
            resp["body"] = result.get("body", "")
            resp["cta"] = result.get("cta", "open_ended")
        elif result["action"] == "wait":
            resp["wait_seconds"] = result.get("wait_seconds", 1800)
        return resp

    except Exception as e:
        return {"action": "send", "body": "Noted — I'll follow up on this shortly.",
                "cta": "none", "rationale": f"Fallback reply (error: {str(e)[:80]})"}
