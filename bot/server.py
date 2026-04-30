"""
Vera Message Engine — FastAPI Server
All 5 endpoints for the magicpin AI Challenge judge harness.
"""
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from bot.config import (TEAM_NAME, TEAM_MEMBERS, MODEL_NAME, APPROACH,
                        CONTACT_EMAIL, BOT_VERSION, MAX_ACTIONS_PER_TICK)
from bot.context_store import ContextStore
from bot.composer import compose
from bot.reply_handler import handle_reply
from bot.prioritizer import prioritize_triggers

# ─── Load env ─────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Vera Message Engine", version=BOT_VERSION)
START_TIME = time.time()
store = ContextStore()


# ─── Request/Response Models ─────────────────────────────────────────────────

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint — shows bot info."""
    return {
        "name": "Vera Message Engine",
        "version": BOT_VERSION,
        "status": "running",
        "endpoints": [
            "GET  /v1/healthz",
            "GET  /v1/metadata",
            "POST /v1/context",
            "POST /v1/tick",
            "POST /v1/reply",
        ]
    }


@app.get("/v1/healthz")
async def healthz():
    """Liveness probe — returns context counts and uptime."""
    counts = store.count_by_scope()
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": {
            "category": counts.get("category", 0),
            "merchant": counts.get("merchant", 0),
            "customer": counts.get("customer", 0),
            "trigger": counts.get("trigger", 0),
        }
    }


@app.get("/v1/metadata")
async def metadata():
    """Bot identity and approach."""
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": MODEL_NAME,
        "approach": APPROACH,
        "contact_email": CONTACT_EMAIL,
        "version": BOT_VERSION,
        "submitted_at": "2026-04-29T08:00:00Z",
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    """
    Receive a context push. Idempotent by (context_id, version).
    Higher version replaces atomically.
    """
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return {"accepted": False, "reason": "invalid_scope",
                "details": f"Scope must be one of {valid_scopes}"}

    accepted, reason, current_version = store.upsert(
        body.scope, body.context_id, body.version, body.payload
    )

    if accepted:
        return {
            "accepted": True,
            "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": datetime.now(timezone.utc).isoformat() + "Z",
        }
    else:
        return {
            "accepted": False,
            "reason": reason,
            "current_version": current_version,
        }


@app.post("/v1/tick")
async def tick(body: TickBody):
    """
    Periodic wake-up. Bot inspects context and decides what to send.
    Returns list of proactive message actions.
    """
    if not body.available_triggers:
        return {"actions": []}

    # Prioritize triggers
    ranked = prioritize_triggers(store, body.available_triggers, body.now)

    actions = []
    for item in ranked:
        if len(actions) >= MAX_ACTIONS_PER_TICK:
            break

        trigger = item["trigger"]
        merchant = item["merchant"]
        category = item["category"]
        customer = item.get("customer")
        merchant_id = item["merchant_id"]
        trigger_id = item["trigger_id"]
        customer_id = item.get("customer_id")

        # Compose the message
        try:
            result = compose(category, merchant, trigger, customer)
        except Exception as e:
            continue

        body_text = result.get("body", "")
        if not body_text:
            continue

        # Generate conversation ID
        kind = trigger.get("kind", "msg")
        conv_id = f"conv_{merchant_id}_{trigger_id}"

        # Check anti-repetition
        if store.is_repeated(conv_id, body_text):
            continue

        # Record the send
        store.record_sent(conv_id, body_text)
        store.suppress(result.get("suppression_key", ""), 86400)

        # Record conversation turn
        store.add_conversation_turn(conv_id, {
            "from_role": "vera",
            "body": body_text,
            "trigger_id": trigger_id,
            "merchant_id": merchant_id,
            "ts": body.now,
        })

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": result.get("send_as", "vera"),
            "trigger_id": trigger_id,
            "template_name": result.get("template_name", f"vera_{kind}_v1"),
            "template_params": result.get("template_params", []),
            "body": body_text,
            "cta": result.get("cta", "open_ended"),
            "suppression_key": result.get("suppression_key", ""),
            "rationale": result.get("rationale", ""),
        })

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    """
    Receive a reply from simulated merchant/customer.
    Returns the bot's next action.
    """
    result = handle_reply(
        store=store,
        conversation_id=body.conversation_id,
        merchant_id=body.merchant_id or "",
        customer_id=body.customer_id,
        from_role=body.from_role,
        message=body.message,
        turn_number=body.turn_number,
    )

    # Record bot's response in conversation
    if result.get("action") == "send" and result.get("body"):
        store.add_conversation_turn(body.conversation_id, {
            "from_role": "vera",
            "body": result["body"],
            "merchant_id": body.merchant_id,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        store.record_sent(body.conversation_id, result["body"])

    return result


# ─── Optional teardown ────────────────────────────────────────────────────────

@app.post("/v1/teardown")
async def teardown():
    """Wipe state at end of test."""
    global store
    store = ContextStore()
    return {"status": "wiped"}
