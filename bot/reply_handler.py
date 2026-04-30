"""
Reply handler — multi-turn conversation intelligence.
Handles auto-reply detection, intent transitions, hostile exits, and off-topic redirection.
"""
from typing import Optional
from bot.config import (AUTO_REPLY_PATTERNS, COMMITMENT_PHRASES,
                        HOSTILE_PHRASES, OFF_TOPIC_PHRASES)
from bot.context_store import ContextStore
from bot.composer import compose_reply


def detect_auto_reply(message: str) -> bool:
    """Detect if a message is a WhatsApp Business auto-reply."""
    msg_lower = message.lower().strip()
    return any(pattern in msg_lower for pattern in AUTO_REPLY_PATTERNS)


def detect_commitment(message: str) -> bool:
    """Detect if the merchant is committing / saying yes."""
    msg_lower = message.lower().strip()
    # Exact short matches
    if msg_lower in ("ok", "yes", "ya", "yep", "yup", "sure", "haan", "confirm", "approved", "go", "do it"):
        return True
    return any(phrase in msg_lower for phrase in COMMITMENT_PHRASES)


def detect_hostile(message: str) -> bool:
    """Detect hostile / opt-out messages."""
    msg_lower = message.lower().strip()
    return any(phrase in msg_lower for phrase in HOSTILE_PHRASES)


def detect_off_topic(message: str) -> bool:
    """Detect off-topic requests."""
    msg_lower = message.lower().strip()
    return any(phrase in msg_lower for phrase in OFF_TOPIC_PHRASES)


def count_auto_replies_in_conversation(store: ContextStore, conversation_id: str) -> int:
    """Count consecutive auto-replies at the end of conversation."""
    turns = store.get_conversation(conversation_id)
    count = 0
    for turn in reversed(turns):
        if turn.get("from_role") == "merchant" and turn.get("is_auto_reply"):
            count += 1
        elif turn.get("from_role") == "merchant":
            break
    return count


def handle_reply(store: ContextStore, conversation_id: str, merchant_id: str,
                 customer_id: Optional[str], from_role: str, message: str,
                 turn_number: int) -> dict:
    """
    Handle an incoming reply from merchant or customer.
    Returns the bot's response action.
    """
    # Record this turn
    store.add_conversation_turn(conversation_id, {
        "from_role": from_role,
        "body": message,
        "turn_number": turn_number,
        "merchant_id": merchant_id,
    })

    # Check if conversation was already ended
    if store.is_conversation_ended(conversation_id):
        return {
            "action": "end",
            "rationale": "Conversation was previously ended."
        }

    # ─── Auto-reply detection ─────────────────────────────────────────
    if detect_auto_reply(message):
        store.add_conversation_turn(conversation_id, {
            "from_role": "system", "is_auto_reply": True,
            "body": message, "merchant_id": merchant_id,
        })
        auto_count = count_auto_replies_in_conversation(store, conversation_id)

        if auto_count >= 3:
            store.end_conversation(conversation_id, "Auto-reply 3x — closing")
            return {
                "action": "end",
                "rationale": "Auto-reply detected 3+ times in a row. No real engagement signal; closing conversation."
            }
        elif auto_count >= 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Same auto-reply twice in a row — owner not at phone. Wait 24h before retry."
            }
        else:
            return {
                "action": "send",
                "body": "Looks like an auto-reply — when the owner sees this, just reply 'Yes' and I'll take it from there.",
                "cta": "binary_yes_no",
                "rationale": "Detected auto-reply; one explicit prompt to flag it for the owner."
            }

    # ─── Hostile / opt-out detection ──────────────────────────────────
    if detect_hostile(message):
        store.end_conversation(conversation_id, "Merchant hostile/opt-out")
        store.suppress_merchant(merchant_id, days=30)
        return {
            "action": "send",
            "body": "Apologies — I won't message again. If anything changes, you can always restart with 'Hi Vera'.",
            "cta": "none",
            "rationale": "Merchant explicitly opted out. Sending one-line acknowledgment + suppressing all triggers for 30 days."
        }

    # ─── Off-topic detection ──────────────────────────────────────────
    if detect_off_topic(message):
        # Get context for redirect
        merchant = store.get_merchant(merchant_id)
        original_trigger = _find_original_trigger(store, conversation_id)
        trigger_kind = original_trigger.get("kind", "your update") if original_trigger else "your update"

        return {
            "action": "send",
            "body": f"I'll have to leave that to your CA/consultant — that's outside what I can help with directly. Coming back to {trigger_kind} — shall I continue with what we were working on?",
            "cta": "open_ended",
            "rationale": "Out-of-scope ask politely declined; redirecting back to original trigger."
        }

    # ─── Intent/commitment detection ──────────────────────────────────
    if detect_commitment(message):
        merchant = store.get_merchant(merchant_id) or {}
        category = store.get_category_for_merchant(merchant) or {}
        original_trigger = _find_original_trigger(store, conversation_id) or {}
        customer = store.get_customer(customer_id) if customer_id else None
        conv_history = store.get_conversation(conversation_id)

        # Use LLM to compose action-mode response
        result = compose_reply(category, merchant, original_trigger,
                               conv_history, message, customer)

        # Ensure it's in action mode, not qualifying
        if result.get("action") == "send":
            body = result.get("body", "")
            qualifying_words = ["would you", "do you think", "can you tell me", "how about"]
            if any(q in body.lower() for q in qualifying_words):
                # Override with action-mode response
                owner = merchant.get("identity", {}).get("owner_first_name", "")
                result["body"] = (f"Great, {owner}! Working on it now — I'll have the draft ready in a moment. "
                                  f"Reply CONFIRM when you'd like me to proceed.")
                result["cta"] = "binary_confirm_cancel"
                result["rationale"] = "Merchant committed; switching to action mode immediately."

        return result

    # ─── General reply — use LLM ─────────────────────────────────────
    merchant = store.get_merchant(merchant_id) or {}
    category = store.get_category_for_merchant(merchant) or {}
    original_trigger = _find_original_trigger(store, conversation_id) or {}
    customer = store.get_customer(customer_id) if customer_id else None
    conv_history = store.get_conversation(conversation_id)

    return compose_reply(category, merchant, original_trigger,
                         conv_history, message, customer)


def _find_original_trigger(store: ContextStore, conversation_id: str) -> Optional[dict]:
    """Find the original trigger that started a conversation."""
    turns = store.get_conversation(conversation_id)
    for turn in turns:
        trigger_id = turn.get("trigger_id")
        if trigger_id:
            trigger = store.get_trigger(trigger_id)
            if trigger:
                return trigger
    return None
