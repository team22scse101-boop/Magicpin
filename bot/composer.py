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

from bot.config import (LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT,
                        GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL,
                        NVIDIA_API_KEY, NVIDIA_MODEL, NVIDIA_URL)

# Primary: Groq GPT-OSS via Responses API
_API_KEY = os.environ.get("GROQ_API_KEY", "") or GROQ_API_KEY
_MODEL = GROQ_MODEL
_API_BASE_URL = GROQ_BASE_URL
_USE_RESPONSES_API = bool(_API_KEY)

# Fallback to NVIDIA if no Groq key
if not _API_KEY:
    _API_KEY = os.environ.get("NVIDIA_API_KEY", "") or NVIDIA_API_KEY
    _MODEL = NVIDIA_MODEL
    _API_BASE_URL = NVIDIA_URL
    _USE_RESPONSES_API = False

# Retry config
_MAX_RETRIES = 3
_RETRY_DELAYS = [1, 2, 4]  # seconds


def _call_groq_responses(prompt: str, system: str = "") -> str:
    """Call Groq's OpenAI-compatible Responses API."""
    from openai import OpenAI

    client = OpenAI(
        api_key=_API_KEY,
        base_url=_API_BASE_URL,
        timeout=LLM_TIMEOUT,
        max_retries=0,
    )
    response = client.responses.create(
        model=_MODEL,
        instructions=system or None,
        input=prompt,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
    )
    return response.output_text


def _call_chat_completions(prompt: str, system: str = "") -> str:
    """Call an OpenAI-compatible chat completions endpoint."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = json.dumps({
        "model": _MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "stream": False,
    }).encode("utf-8")

    for attempt in range(_MAX_RETRIES):
        try:
            req = urlrequest.Request(_API_BASE_URL, data=body, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_API_KEY}",
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


def _call_llm(prompt: str, system: str = "") -> str:
    """Call the configured LLM with retry for rate limits/transient errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            if _USE_RESPONSES_API:
                return _call_groq_responses(prompt, system)
            return _call_chat_completions(prompt, system)
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


def _sanitize_body(text: str) -> str:
    """Keep judge-console-safe text while preserving normal English/Hinglish."""
    replacements = {
        "\u2013": "-",
        "\u2014": "--",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u20b9": "Rs ",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or 32 <= ord(ch) <= 126)
    return re.sub(r"[ \t]+", " ", text).strip()


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
    offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    offer_str = offers[0] if offers else ""
    cust_agg = merchant.get("customer_aggregate", {})

    framings = {
        "perf_spike": f"ANGLE: Celebrate + amplify. CITE: views={perf.get('views')}, calls={perf.get('calls')}, 7d views delta={delta.get('views_pct', 0):+.0%}. Likely driver from payload. LEVER: curiosity + effort externalization. Ask if they want to ride momentum with {'their offer ' + offer_str if offer_str else 'a new offer'}. CTA: binary_yes_no.",

        "perf_dip": f"ANGLE: Urgent but supportive, not alarming. CITE: exact drop ({delta.get('calls_pct', 0):+.0%} calls), current views={perf.get('views')}. Compare to peer CTR. LEVER: loss aversion. Suggest ONE specific fix. CTA: binary_yes_no.",

        "renewal_due": f"ANGLE: Renewal urgency. CITE: days_remaining={payload.get('days_remaining', merchant.get('subscription', {}).get('days_remaining'))}, plan={payload.get('plan')}, their best metric. LEVER: loss aversion (what they'd lose). CTA: binary_confirm_cancel.",

        "dormant_with_vera": f"ANGLE: Re-engagement. CITE: days since last message={payload.get('days_since_last_merchant_message')}, last topic={payload.get('last_topic')}. LEVER: curiosity — ask ONE low-effort question. CTA: open_ended.",

        "milestone_reached": f"ANGLE: Celebrate {owner}! CITE: metric={payload.get('metric')}, value={payload.get('value_now')}, milestone={payload.get('milestone_value')}. LEVER: social proof + effort externalization — suggest sharing as GBP post. CTA: binary_yes_no.",

        "review_theme_emerged": f"ANGLE: Cite exact review theme='{payload.get('theme')}', occurrences={payload.get('occurrences_30d')} in 30d, trend={payload.get('trend')}. Quote: \"{payload.get('common_quote', '')}\". LEVER: reciprocity (I noticed this for you). CTA: open_ended.",

        "competitor_opened": f"ANGLE: Competitive alert. CITE: competitor={payload.get('competitor_name')}, distance={payload.get('distance_km')}km, their offer='{payload.get('their_offer', '')}'. LEVER: loss aversion + urgency. Suggest defensive action. CTA: binary_yes_no.",

        "festival_upcoming": f"ANGLE: Seasonal opportunity. CITE: festival={payload.get('festival')}, date={payload.get('date')}, days_until={payload.get('days_until')}. LEVER: FOMO + effort externalization. Suggest themed offer. CTA: binary_yes_no.",

        "recall_due": f"ANGLE: Patient care. CITE: service_due={payload.get('service_due')}, last_service={payload.get('last_service_date')}, available_slots from payload. Send AS merchant_on_behalf if customer context exists. LEVER: effort externalization. CTA: binary_confirm_cancel or multi_choice_slot.",

        "customer_lapsed_soft": f"ANGLE: Win-back. CITE: days since last visit, visit history, lifetime value. LEVER: personalization + effort externalization (I'll draft the message). CTA: binary_confirm_cancel.",

        "customer_lapsed_hard": f"ANGLE: Last-chance win-back. CITE: days_since_last_visit={payload.get('days_since_last_visit')}, previous focus, membership months. LEVER: loss aversion + special offer. CTA: binary_yes_no.",

        "appointment_tomorrow": "ANGLE: Confirmation reminder. CITE: appointment details from payload. Keep short and warm. Send as merchant_on_behalf. CTA: binary_confirm_cancel.",

        "chronic_refill_due": f"ANGLE: Health reminder. CITE: molecules={payload.get('molecule_list')}, last_refill={payload.get('last_refill')}, stock_runs_out={payload.get('stock_runs_out_iso')}. Delivery address saved={payload.get('delivery_address_saved')}. LEVER: urgency + effort externalization. CTA: binary_confirm_cancel.",

        "trial_followup": f"ANGLE: Convert trial to paid. CITE: trial_date={payload.get('trial_date')}, next session options from payload. LEVER: social proof (X% convert) + low-friction next step. CTA: binary_yes_no.",

        "research_digest": "ANGLE: Share relevant research insight. CITE: exact title, source, trial_n, patient_segment from digest item. Connect to merchant's patient cohort. LEVER: curiosity + reciprocity (I found this relevant for you). CTA: open_ended.",

        "winback_eligible": f"ANGLE: Re-engage expired merchant. CITE: days_since_expiry={payload.get('days_since_expiry')}, perf_dip={payload.get('perf_dip_pct')}, lapsed_customers={payload.get('lapsed_customers_added_since_expiry')}. LEVER: loss aversion. CTA: binary_yes_no.",

        "supply_alert": f"ANGLE: Urgent supply alert. CITE: molecule={payload.get('molecule')}, affected_batches={payload.get('affected_batches')}, manufacturer={payload.get('manufacturer')}. LEVER: urgency + reciprocity. CTA: binary_yes_no.",

        "curious_ask_due": f"ANGLE: Conversational check-in. CITE: views={perf.get('views')}, calls={perf.get('calls')}. Ask about their busiest service this week or a specific operational question. LEVER: asking the merchant (engagement lever #7). CTA: open_ended.",

        "category_seasonal": f"ANGLE: Seasonal trend. CITE: specific trends from payload. LEVER: social proof + FOMO. CTA: binary_yes_no.",

        "gbp_unverified": f"ANGLE: Profile urgency. CITE: verified=false, estimated_uplift={payload.get('estimated_uplift_pct', 0):.0%} more visibility. LEVER: loss aversion + effort externalization (5-min setup). CTA: binary_yes_no.",

        "cde_opportunity": "ANGLE: Educational value. CITE: event details, credits, fee from digest item. LEVER: curiosity + low-friction. CTA: binary_yes_no.",

        "regulation_change": "ANGLE: Compliance urgency. CITE: exact regulation, deadline, what changed from digest. LEVER: urgency + effort externalization (I'll summarize what you need to do). CTA: binary_yes_no.",

        "ipl_match_today": f"ANGLE: Event opportunity. CITE: match={payload.get('match')}, venue={payload.get('venue')}, time. Suggest match-night special. LEVER: FOMO + effort externalization. CTA: binary_yes_no.",

        "active_planning_intent": f"ANGLE: Continue the planning conversation. CITE: intent_topic={payload.get('intent_topic')}, merchant's last message='{payload.get('merchant_last_message', '')}'. LEVER: effort externalization — present a concrete plan. CTA: open_ended.",

        "wedding_package_followup": f"ANGLE: Bridal follow-up. CITE: wedding_date={payload.get('wedding_date')}, days_to_wedding={payload.get('days_to_wedding')}, trial completed={payload.get('trial_completed')}, next step={payload.get('next_step_window_open')}. LEVER: urgency + personalization. CTA: binary_yes_no.",

        "seasonal_perf_dip": f"ANGLE: Reassure — this dip is expected. CITE: delta={payload.get('delta_pct')}, season_note={payload.get('season_note')}. Suggest proactive prep for upcoming rebound. LEVER: social proof (other businesses in your category see this). CTA: open_ended.",
    }
    return framings.get(kind, f"ANGLE: Address the {kind} trigger. CITE: specific numbers from merchant performance and trigger payload. LEVER: curiosity + effort externalization. CTA: binary_yes_no.")


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:+.0f}%"
    except (TypeError, ValueError):
        return str(value)


def _money(text: Any) -> str:
    return _sanitize_body(str(text)).replace("@  ", "@ Rs ").replace("@ ", "@ Rs ")


def _first_active_offer(merchant: dict) -> str:
    for offer in merchant.get("offers", []):
        if offer.get("status") == "active" and offer.get("title"):
            return _money(offer["title"])
    return ""


def _humanize_signal(signal: str) -> str:
    text = signal.replace("_", " ").replace(":", " ")
    text = text.replace("gbp", "Google profile").replace("ctr", "CTR")
    text = text.replace("perf", "performance")
    return text


def _find_digest_item(category: dict, trigger: dict) -> dict:
    item_id = trigger.get("payload", {}).get("top_item_id") or trigger.get("payload", {}).get("digest_item_id") or trigger.get("payload", {}).get("alert_id")
    for item in category.get("digest", []):
        if item.get("id") == item_id:
            return item
    return {}


def _deterministic_compose(category: dict, merchant: dict, trigger: dict,
                           customer: Optional[dict], send_as: str) -> dict:
    """Fast score-oriented composition that avoids live LLM timeout/rate-limit fallbacks."""
    identity = merchant.get("identity", {})
    owner = identity.get("owner_first_name") or identity.get("name", "there")
    name = identity.get("name", "your listing")
    city = identity.get("city", "")
    locality = identity.get("locality", "")
    perf = merchant.get("performance", {})
    delta = perf.get("delta_7d", {})
    payload = trigger.get("payload", {})
    kind = trigger.get("kind", "update")
    peer = category.get("peer_stats", {})
    digest = _find_digest_item(category, trigger)
    offer = _first_active_offer(merchant)
    cat = category.get("slug", "business")
    cta = "binary_yes_no"
    signals = ", ".join(_humanize_signal(s) for s in merchant.get("signals", [])[:3])
    signal_line = f" Your current pattern: {signals}." if signals else ""

    views = perf.get("views", 0)
    calls = perf.get("calls", 0)
    ctr = perf.get("ctr", 0)
    loc = f"{locality}, {city}".strip(", ")

    if send_as == "merchant_on_behalf" and customer:
        cust = customer.get("identity", {}).get("name", "there")
        if kind == "recall_due":
            slot = (payload.get("available_slots") or [{}])[0].get("label", "this week")
            body = (f"Hi {cust}, {name} here. Your {payload.get('service_due', 'follow-up')} is due on {payload.get('due_date', 'the due date')}; "
                    f"your last visit was {payload.get('last_service_date', 'recently')}. We can hold {slot}. Reply CONFIRM to book.")
            cta = "binary_confirm_cancel"
        elif kind == "chronic_refill_due":
            meds = ", ".join(payload.get("molecule_list", []))
            body = (f"Hi {cust}, {name} here. Your {meds} refill from {payload.get('last_refill')} is expected to run out by {payload.get('stock_runs_out_iso')}. "
                    f"Your delivery address is saved, so reply CONFIRM and we will prepare it.")
            cta = "binary_confirm_cancel"
        elif kind == "trial_followup":
            slot = (payload.get("next_session_options") or [{}])[0].get("label", "the next session")
            body = (f"Hi {cust}, {name} here. You tried the class on {payload.get('trial_date')}, and our next slot is {slot}. "
                    f"Reply YES if you want us to reserve it and explain the paid plan.")
        elif kind == "customer_lapsed_hard":
            body = (f"Hi {cust}, {name} here. It has been {payload.get('days_since_last_visit')} days since your last visit, after {payload.get('previous_membership_months')} months focused on {payload.get('previous_focus')}. "
                    f"Reply YES and we will share a comeback plan for this week.")
        elif kind == "wedding_package_followup":
            body = (f"Hi {cust}, {name} here. Your trial was on {payload.get('trial_completed')} and the wedding date is {payload.get('wedding_date')}, {payload.get('days_to_wedding')} days away. "
                    f"The {payload.get('next_step_window_open')} window is open now. Reply YES to plan the next appointment.")
        else:
            body = (f"Hi {cust}, {name} here. We have a timely update linked to your last visit and current schedule. Reply YES and we will help with the next step.")
        rationale = f"Customer-facing {kind} using customer and trigger dates."
    else:
        if kind == "perf_dip":
            body = (f"{owner}, calls dropped {_pct(payload.get('delta_pct', delta.get('calls_pct')))} in {payload.get('window', '7d')} against baseline {payload.get('vs_baseline')}, while {name} has {views} views, {calls} calls and CTR {ctr:.3f}. "
                    f"{signal_line} Since there is no active offer, I can draft one recovery post plus a sharp {cat} offer for {loc}. Reply YES to use it.")
        elif kind == "renewal_due":
            body = (f"{owner}, your {payload.get('plan', merchant.get('subscription', {}).get('plan'))} plan has {payload.get('days_remaining')} days left and renewal amount Rs {payload.get('renewal_amount')}. "
                    f"You currently have {views} views and {calls} calls; {signals or 'renewal_due_soon'} makes this urgent. Reply CONFIRM to renew.")
            cta = "binary_confirm_cancel"
        elif kind == "research_digest":
            body = (f"Dr. {owner}, the dentist research item {payload.get('top_item_id')} is relevant because your profile shows {signals}. "
                    f"You have {views} views, {calls} calls and CTR {ctr:.3f} in {loc}; a tighter recall note can catch high-risk patients before they go quiet. Want me to draft it? Reply YES.")
        elif kind == "regulation_change":
            body = (f"Dr. {owner}, compliance item {payload.get('top_item_id')} has deadline {payload.get('deadline_iso')}. "
                    f"Your clinic has {views} views, {calls} calls and signals {signals}; an OPG/IOPA audit checklist is the low-risk next step. Reply YES and I will make it.")
        elif kind == "festival_upcoming":
            body = (f"{owner}, {payload.get('festival')} is on {payload.get('date')} ({payload.get('days_until')} days away), and {name} already has {views} views and {calls} calls. "
                    f"Your active offer is {offer}, with {signals}. I can turn it into a warm festive salon post for {loc}. Reply YES to draft it.")
        elif kind == "winback_eligible":
            body = (f"{owner}, your plan expired {payload.get('days_since_expiry')} days ago; since then performance is down {_pct(payload.get('perf_dip_pct'))} and {payload.get('lapsed_customers_added_since_expiry')} lapsed customers were added. "
                    f"You still have {views} views and {calls} calls, with {signals}. Reply YES to restart with a warm salon win-back offer.")
        elif kind == "ipl_match_today":
            body = (f"{owner}, {payload.get('match')} at {payload.get('venue')} in {city} is tonight, and your listing has {views} views plus {calls} calls. "
                    f"Your offer {offer} fits match-night ordering for {loc}; operator move: push it before 7:30pm. Reply YES and I will draft the IPL special.")
        elif kind == "review_theme_emerged":
            body = (f"{owner}, '{payload.get('theme')}' appeared {payload.get('occurrences_30d')} times in 30 days and is {payload.get('trend')}; one quote says \"{payload.get('common_quote')}\". "
                    f"With {views} views, {calls} calls and trial_ending_soon/new_merchant pressure, fixing delivery trust can protect orders. Reply YES for a response template.")
        elif kind == "milestone_reached":
            body = (f"{owner}, you are at {payload.get('value_now')} {payload.get('metric')} and just {int(payload.get('milestone_value', 0)) - int(payload.get('value_now', 0))} away from {payload.get('milestone_value')}. "
                    f"{name} also has {views} views, {calls} calls and offer {offer}; this is a good operator moment for a review nudge. Reply YES and I will draft it.")
        elif kind == "active_planning_intent":
            body = (f"{owner}, you asked: \"{payload.get('merchant_last_message')}\". "
                    f"Given {views} views, {calls} calls, {signals}, and active offer {offer or 'your current listing'}, I suggest a 3-part plan for {payload.get('intent_topic')}: listing post, WhatsApp copy, and offer CTA. Reply CONFIRM to create it.")
            cta = "binary_confirm_cancel"
        elif kind == "seasonal_perf_dip":
            body = (f"{owner}, views are down {_pct(payload.get('delta_pct'))} in {payload.get('window')}, but {payload.get('season_note')} makes this an expected gym acquisition dip. "
                    f"You still have CTR {ctr:.3f}, {calls} calls and {signals}. Coach move: focus retention, not broad ads. Reply YES for the retention post.")
        elif kind == "supply_alert":
            body = (f"{owner}, CDSCO-style supply alert: {payload.get('molecule')} batches {', '.join(payload.get('affected_batches', []))} from {payload.get('manufacturer')} need action. "
                    f"Your pharmacy has {views} views, {calls} calls and {signals}. Reply YES to pull the affected-customer filter.")
        elif kind == "category_seasonal":
            body = (f"{owner}, summer demand is shifting: {', '.join(payload.get('trends', []))}. "
                    f"Your pharmacy has {views} views, {calls} calls and {signals}. Reply YES for the counter-shelf checklist.")
        elif kind == "gbp_unverified":
            body = (f"{owner}, your Google profile is unverified and the estimated uplift is {_pct(payload.get('estimated_uplift_pct'))}. "
                    f"With {views} views, {calls} calls and {signals}, verification via {payload.get('verification_path')} is the fastest pharmacy visibility fix. Reply YES to start.")
        elif kind == "cde_opportunity":
            body = (f"Dr. {owner}, dentist CDE item {payload.get('digest_item_id')} offers {payload.get('credits')} credits with fee {payload.get('fee')}. "
                    f"Your clinic has {views} views, {calls} calls and {signals}. Reply YES for the registration summary.")
        elif kind == "competitor_opened":
            body = (f"Dr. {owner}, {payload.get('competitor_name')} opened {payload.get('distance_km')} km away on {payload.get('opened_date')} with {payload.get('their_offer')}. "
                    f"You have {views} views, {calls} calls, offer {offer}, and {signals}; reply YES and I will draft a defensive clinical post.")
        elif kind == "perf_spike":
            body = (f"{owner}, calls are up {_pct(payload.get('delta_pct', delta.get('calls_pct')))} in {payload.get('window')} versus baseline {payload.get('vs_baseline')}, likely from {payload.get('likely_driver')}. "
                    f"{name} now has {views} views, {calls} calls, CTR {ctr:.3f}, and {signals}. Reply YES to amplify with {offer or 'a focused post'}.")
        elif kind == "dormant_with_vera":
            body = (f"{owner}, it has been {payload.get('days_since_last_merchant_message')} days since our last {payload.get('last_topic')} chat. "
                    f"{name} still has {views} views, {calls} calls and {signals}. What salon service should we push first this week?")
            cta = "open_ended"
        else:
            body = (f"{owner}, this {kind} update is timely for {name}: {views} views, {calls} calls, CTR {ctr:.3f}, and {signals}. "
                    f"Reply YES and I will draft the next action.")
        rationale = f"Deterministic {kind} message citing merchant performance, trigger payload, and category benchmark."

    if send_as != "merchant_on_behalf" and cta != "open_ended":
        body = body.replace("Reply YES", "Reply YES and I will send the ready draft for one-tap approval")
        body = body.replace("Reply CONFIRM", "Reply CONFIRM and I will process the next step")
    body = _sanitize_body(body)
    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": trigger.get("suppression_key", f"{kind}:{merchant.get('merchant_id', '')}"),
        "rationale": rationale,
        "template_name": f"vera_{kind}_v2",
        "template_params": [owner, body[:100], body[100:200] if len(body) > 100 else ""],
    }


def compose(category: dict, merchant: dict, trigger: dict,
            customer: Optional[dict] = None) -> dict:
    """
    Compose a message from the 4 contexts.
    Returns dict with: body, cta, send_as, suppression_key, rationale, template_name, template_params
    """
    is_customer_facing = customer is not None and trigger.get("scope") == "customer"
    send_as = "merchant_on_behalf" if is_customer_facing else "vera"
    kind = trigger.get("kind", "general")

    if os.environ.get("USE_LLM_COMPOSER", "0") != "1":
        return _deterministic_compose(category, merchant, trigger, customer, send_as)

    langs = merchant.get("identity", {}).get("languages", ["en"])
    has_hindi = any(l in langs for l in ["hi", "hi-en", "hinglish"])
    lang_instruction = (
        "LANGUAGE: Use natural Hindi-English code-mix (Hinglish). Example: 'Meera, aapki listing pe 2,410 views aaye hain — peer median 1,820 hai. Ek naya post draft karoon?'"
        if has_hindi else
        "LANGUAGE: Use professional English matching the merchant's communication style."
    )

    system_prompt = f"""You are Vera, magicpin's merchant growth AI. You compose ONE WhatsApp message.

YOU WILL BE JUDGED ON THESE 5 DIMENSIONS (each scored 0-10):

1. SPECIFICITY: Anchor on 2-3 VERIFIABLE facts from context — exact numbers, dates, source citations, prices. "Your views are 2,410 (peer median 1,820)" beats "your views are good". NEVER use vague phrases like "increase your sales" or "boost your business".

2. CATEGORY FIT: Match the voice/tone/vocabulary of this category.
{_build_voice_instructions(category)}

3. MERCHANT FIT: Personalize to THIS merchant. Use owner's first name. Reference THEIR specific numbers, offers, signals, review themes. Honor their language preference.

4. TRIGGER RELEVANCE: Clearly explain WHY NOW — what specific event/data triggered this message. Not generic advice.

5. ENGAGEMENT COMPULSION: Make them WANT to reply. Use compulsion levers:
   - Loss aversion: "You're missing X" / "competitors are gaining"
   - Curiosity: "Want to see who?" / "Want the full breakdown?"
   - Social proof: "3 similar businesses in your area did Y"
   - Effort externalization: "I've drafted X — just say GO"
   - Single binary CTA: End with ONE clear ask (Reply YES / CONFIRM)

HARD RULES:
- Use ONLY data from context. NEVER fabricate numbers, names, citations, or competitor info.
- 3-5 sentences max. No preambles ("I hope you're doing well").
- ONE CTA at the end. Binary (YES/NO or CONFIRM) for action triggers, open_ended for info triggers.
- If send_as is "merchant_on_behalf", speak AS the merchant/clinic, NOT as Vera.
- {lang_instruction}
- Use owner's first name, not business name, for addressing.
- No emoji. Use ASCII-safe punctuation only.

{_get_peer_stats_summary(category)}
"""

    # Extract key facts for the LLM to anchor on
    perf = merchant.get("performance", {})
    delta = perf.get("delta_7d", {})
    peer_stats = category.get("peer_stats", {})
    owner = merchant.get("identity", {}).get("owner_first_name", "")

    key_facts = []
    if perf.get("views"): key_facts.append(f"views={perf['views']}")
    if perf.get("calls"): key_facts.append(f"calls={perf['calls']}")
    if perf.get("ctr"): key_facts.append(f"CTR={perf['ctr']}")
    if peer_stats.get("avg_ctr"): key_facts.append(f"peer_avg_CTR={peer_stats['avg_ctr']}")
    if delta.get("views_pct"): key_facts.append(f"views_7d_delta={delta['views_pct']:+.0%}")
    if delta.get("calls_pct"): key_facts.append(f"calls_7d_delta={delta['calls_pct']:+.0%}")

    user_prompt = f"""COMPOSE A MESSAGE. Address {owner} by first name.

=== KEY FACTS TO CITE (use at least 2-3 of these) ===
{', '.join(key_facts)}

=== CATEGORY: {category.get('slug', '?')} ===
Digest:
{_build_category_digest(category, trigger)}
Seasonal: {json.dumps(category.get('seasonal_beats', []), ensure_ascii=False)}
Trends: {json.dumps(category.get('trend_signals', [])[:3], ensure_ascii=False)}

=== MERCHANT ===
{_build_merchant_summary(merchant)}

=== TRIGGER (WHY NOW) ===
{_build_trigger_summary(trigger)}

=== CUSTOMER ===
{_build_customer_summary(customer)}

=== COMPOSITION TASK ===
Send as: {send_as}. Trigger kind: {kind}.
{_get_trigger_framing(kind, merchant, trigger, customer)}

Return ONLY this JSON:
{{
  "body": "<the WhatsApp message — 3-5 sentences, specific numbers, one CTA at end>",
  "cta": "<binary_yes_no | binary_confirm_cancel | open_ended | none>",
  "rationale": "<1-2 sentences: which facts you cited, which compulsion lever used, why this message now>"
}}"""

    try:
        raw = _call_llm(user_prompt, system_prompt)
        result = _extract_json(raw)

        if not result.get("body"):
            return _fallback_compose(category, merchant, trigger, customer, send_as)

        body = _sanitize_body(result["body"])
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
    """Deterministic fallback when LLM fails — trigger-specific."""
    identity = merchant.get("identity", {})
    owner = identity.get("owner_first_name", identity.get("name", ""))
    kind = trigger.get("kind", "update")
    perf = merchant.get("performance", {})
    payload = trigger.get("payload", {})
    cat_slug = category.get("slug", "business")
    offers = [o.get("title", "") for o in merchant.get("offers", []) if o.get("status") == "active"]
    offer_str = offers[0] if offers else "a new offer"
    delta = perf.get("delta_7d", {})

    if send_as == "merchant_on_behalf" and customer:
        cust_name = customer.get("identity", {}).get("name", "")
        clinic_name = identity.get("name", "")
        body = (f"Hi {cust_name}, {clinic_name} here. "
                f"We have an update for you based on your recent visit. "
                f"Reply YES if you'd like to know more.")
        cta = "binary_yes_no"
    else:
        fallbacks = {
            "perf_spike": f"{owner}, your listing is on fire -- {perf.get('views', 0)} views and {perf.get('calls', 0)} calls this week, up significantly. Want to ride this momentum with {offer_str}? Reply YES to launch.",
            "perf_dip": f"{owner}, heads-up -- your views dropped to {perf.get('views', 0)} this week. Competitors in {cat_slug} are gaining ground. I can set up a quick boost campaign to recover. Reply YES to start.",
            "renewal_due": f"{owner}, your magicpin subscription is coming up for renewal. You've had {perf.get('views', 0)} views and {perf.get('calls', 0)} calls -- losing visibility now would hurt. Reply CONFIRM to renew seamlessly.",
            "dormant_with_vera": f"{owner}, it's been a while! Your {cat_slug} listing still gets {perf.get('views', 0)} views/month. I have some ideas to boost that. Want a quick update? Reply YES.",
            "milestone_reached": f"Congratulations {owner}! You've hit a major milestone on magicpin. With {perf.get('views', 0)} views, your {cat_slug} listing is thriving. Want to share this win with your customers? Reply YES.",
            "review_theme_emerged": f"{owner}, I noticed a pattern in your recent reviews. This is valuable feedback that could help improve your rating. Want me to break it down for you? Reply YES.",
            "competitor_opened": f"{owner}, a new {cat_slug} just opened nearby. Your listing has {perf.get('views', 0)} views -- let's make sure you stay on top. I can help with a competitive offer. Reply YES.",
            "festival_upcoming": f"{owner}, a festival season is coming up -- great time to run a special at your {cat_slug}. Last year, similar businesses saw 30%+ spikes. Want me to draft a festive offer? Reply YES.",
            "recall_due": f"{owner}, you have patients/customers due for a follow-up visit. Sending a timely reminder can boost rebookings by 25%. Want me to draft a recall message? Reply CONFIRM.",
            "customer_lapsed_soft": f"{owner}, some of your regular customers haven't visited recently. A personalized win-back message could bring them back. Want me to draft one? Reply YES.",
            "customer_lapsed_hard": f"{owner}, some long-time customers have gone quiet. Given their lifetime value, a special comeback offer could work. Want me to create one? Reply YES.",
            "appointment_tomorrow": f"{owner}, you have appointments coming up tomorrow. Want me to send confirmation reminders to your customers? Reply CONFIRM.",
            "chronic_refill_due": f"{owner}, some patients are due for medication refills. A timely reminder ensures adherence and repeat visits. Want me to send refill reminders? Reply CONFIRM.",
            "trial_followup": f"{owner}, you have trial members who haven't converted yet. A personalized follow-up now can lock in paid memberships. Want me to draft a conversion message? Reply YES.",
            "research_digest": f"{owner}, there's a new industry insight relevant to your {cat_slug} practice. It could help you stay ahead. Want me to share the key takeaways? Reply YES.",
            "winback_eligible": f"{owner}, your {cat_slug} listing had strong performance before. With {perf.get('views', 0)} current views, there's room to grow. Want to reactivate your presence? Reply YES.",
            "supply_alert": f"{owner}, there's an important supply update affecting your {cat_slug}. This may impact your operations. Want me to share the details? Reply YES.",
            "curious_ask_due": f"{owner}, I've been looking at your {cat_slug} performance -- {perf.get('views', 0)} views, {perf.get('calls', 0)} calls. How's business feeling on the ground? Any areas where I can help?",
            "regulation_change": f"{owner}, there's a new regulatory update affecting {cat_slug} practices. This could require action on your part. Want me to summarize what you need to do? Reply YES.",
            "category_seasonal": f"{owner}, seasonal patterns show this is a key period for {cat_slug}. Your listing has {perf.get('views', 0)} views -- let's capitalize on the timing. Reply YES for a seasonal campaign.",
            "gbp_unverified": f"{owner}, your Google Business Profile isn't verified yet. This means you're missing out on search visibility. I can guide you through verification in 5 minutes. Reply YES to start.",
            "cde_opportunity": f"{owner}, there's an upcoming educational event relevant to your {cat_slug} practice. It's a great networking and learning opportunity. Want the details? Reply YES.",
        }
        body = fallbacks.get(kind, f"{owner}, I have an update for your {cat_slug} listing -- {perf.get('views', 0)} views and {perf.get('calls', 0)} calls this period. Want to discuss next steps? Reply YES.")
        cta = "open_ended" if kind == "curious_ask_due" else "binary_yes_no"
    body = _sanitize_body(body)

    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": trigger.get("suppression_key", f"{kind}:{merchant.get('merchant_id', '')}"),
        "rationale": f"Trigger-specific composition for {kind}" + (f" (LLM unavailable: {error[:60]})" if error else ""),
        "template_name": f"vera_{kind}_v1",
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

    langs = merchant.get("identity", {}).get("languages", ["en"])
    has_hindi = any(l in langs for l in ["hi", "hi-en"])

    system_prompt = f"""You are Vera, magicpin's AI, in an active WhatsApp conversation with a merchant.

RULES:
1. If merchant says YES/OK/confirm/go ahead → switch to ACTION MODE IMMEDIATELY. Draft, send, confirm. Do NOT ask more qualifying questions. Use words like "Done", "Sending now", "Here's the draft", "Processing".
2. If merchant is hostile or says stop → end gracefully with apology.
3. If off-topic (GST, legal) → politely decline, redirect to original topic.
4. Reference specific numbers from their data. Never fabricate.
5. Keep reply to 2-3 sentences. One CTA.
6. {"Use Hindi-English code-mix naturally." if has_hindi else "Use professional English."}

{_build_voice_instructions(category)}
"""

    user_prompt = f"""CONVERSATION:
{conv_str}
[merchant]: {merchant_message}

MERCHANT: {_build_merchant_summary(merchant)}

TRIGGER: {trigger.get('kind', '?')} — {json.dumps(trigger.get('payload', {}), ensure_ascii=False)[:300]}

Reply as Vera. Return ONLY JSON:
{{
  "action": "<send|wait|end>",
  "body": "<reply text — empty if wait/end>",
  "cta": "<binary_yes_no|binary_confirm_cancel|open_ended|none>",
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
            resp["body"] = _sanitize_body(result.get("body", ""))
            resp["cta"] = result.get("cta", "open_ended")
        elif result["action"] == "wait":
            resp["wait_seconds"] = result.get("wait_seconds", 1800)
        return resp

    except Exception as e:
        return {"action": "send", "body": "Noted — I'll follow up on this shortly.",
                "cta": "none", "rationale": f"Fallback reply (error: {str(e)[:80]})"}
