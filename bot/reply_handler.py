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
        if turn.get("is_auto_reply"):
            count += 1
        elif turn.get("from_role") in ("merchant", "vera", "system"):
            break
    return count


def handle_reply(store: ContextStore, conversation_id: str, merchant_id: str,
                 customer_id: Optional[str], from_role: str, message: str,
                 turn_number: int) -> dict:
    """
    Handle an incoming reply from merchant or customer.
    Returns the bot's response action.
    """
    # Detect auto-reply early so we can tag the turn
    is_auto = detect_auto_reply(message) if from_role == "merchant" else False

    # Record this turn
    store.add_conversation_turn(conversation_id, {
        "from_role": from_role,
        "body": message,
        "turn_number": turn_number,
        "merchant_id": merchant_id,
        "is_auto_reply": is_auto,
    })

    # If the message is from a customer, route to LLM with customer context
    if from_role == "customer":
        merchant = store.get_merchant(merchant_id) or {}
        category = store.get_category_for_merchant(merchant) or {}
        original_trigger = _find_original_trigger(store, conversation_id) or {}
        customer = store.get_customer(customer_id) if customer_id else {"identity": {"name": "Customer"}}
        conv_history = store.get_conversation(conversation_id)
        return compose_reply(category, merchant, original_trigger,
                             conv_history, message, customer)

    # Check if conversation was already ended
    if store.is_conversation_ended(conversation_id):
        return {
            "action": "end",
            "rationale": "Conversation was previously ended."
        }

    # --- Auto-reply detection ---
    if is_auto:
        auto_count = count_auto_replies_in_conversation(store, conversation_id)
        if auto_count >= 2:
            store.end_conversation(conversation_id, "Auto-reply repeated -- closing")
            return {
                "action": "end",
                "rationale": "Auto-reply detected " + str(auto_count) + " times consecutively. Ending conversation to avoid spam."
            }
        else:
            return {
                "action": "send",
                "body": "Looks like an auto-reply -- when the owner sees this, just reply 'Yes' and I'll take it from there.",
                "cta": "binary_yes_no",
                "rationale": "Detected auto-reply; one explicit prompt to flag it for the owner."
            }

    # --- Hostile / opt-out detection ---
    if detect_hostile(message):
        store.end_conversation(conversation_id, "Merchant hostile/opt-out")
        store.suppress_merchant(merchant_id, days=30)
        return {
            "action": "end",
            "body": "Apologies -- I won't message again. If anything changes, you can always restart with 'Hi Vera'.",
            "cta": "none",
            "rationale": "Merchant explicitly opted out. Ending conversation + suppressing all triggers for 30 days."
        }

    # --- Off-topic detection ---
    if detect_off_topic(message):
        merchant = store.get_merchant(merchant_id)
        original_trigger = _find_original_trigger(store, conversation_id)
        trigger_kind = original_trigger.get("kind", "your update") if original_trigger else "your update"
        return {
            "action": "send",
            "body": "I'll have to leave that to your CA/consultant -- that's outside what I can help with directly. Coming back to " + trigger_kind + " -- shall I continue with what we were working on?",
            "cta": "open_ended",
            "rationale": "Out-of-scope ask politely declined; redirecting back to original trigger."
        }

    # --- Intent/commitment detection ---
    if detect_commitment(message):
        merchant = store.get_merchant(merchant_id) or {}
        category = store.get_category_for_merchant(merchant) or {}
        original_trigger = _find_original_trigger(store, conversation_id) or {}
        customer = store.get_customer(customer_id) if customer_id else None
        conv_history = store.get_conversation(conversation_id)

        # Try LLM first
        result = compose_reply(category, merchant, original_trigger,
                               conv_history, message, customer)

        # Check if we got a real response (not a fallback)
        body = result.get("body", "")
        is_fallback = "follow up" in body.lower() or "noted" in body.lower() or not body

        if result.get("action") == "send" and not is_fallback:
            qualifying_words = ["would you", "do you think", "can you tell me", "how about"]
            if any(q in body.lower() for q in qualifying_words):
                owner = merchant.get("identity", {}).get("owner_first_name", "")
                result["body"] = "Great, " + owner + "! Working on it now -- I'll have the draft ready in a moment. Reply CONFIRM when you'd like me to proceed."
                result["cta"] = "binary_confirm_cancel"
                result["rationale"] = "Merchant committed; switching to action mode immediately."
            return result

        # Smart deterministic fallback based on trigger kind
        owner = merchant.get("identity", {}).get("owner_first_name", "")
        kind = original_trigger.get("kind", "update")
        perf = merchant.get("performance", {})

        action_responses = {
            "perf_dip": "On it, " + owner + "! I'm drafting a visibility boost campaign for you right now. Your views are at " + str(perf.get("views", 0)) + " -- let's push that up. Reply CONFIRM to activate.",
            "perf_spike": "Let's lock this in, " + owner + "! I'll draft a special offer to convert these " + str(perf.get("views", 0)) + " views into more calls. Reply CONFIRM to launch.",
            "renewal_due": "Processing your renewal now, " + owner + ". Your current plan keeps your profile visible to " + str(perf.get("views", 0)) + "+ monthly visitors. Reply CONFIRM to renew.",
            "review_theme_emerged": "Working on it, " + owner + "! I'll draft a response template for those reviews. Reply CONFIRM when you'd like me to publish.",
            "competitor_opened": "Smart move, " + owner + "! I'm preparing a competitive boost package -- updated photos + a new offer to stand out. Reply CONFIRM to activate.",
            "festival_upcoming": "Let's go, " + owner + "! I'll create a festive offer for your listing right away. Reply CONFIRM once you see the draft.",
            "recall_due": "Sending the recall message now, " + owner + "! The patient will get a personalized reminder. Reply CONFIRM to send.",
            "customer_lapsed_soft": "Drafting a win-back message now, " + owner + "! I'll include their visit history to make it personal. Reply CONFIRM to send.",
            "dormant_with_vera": "Great to have you back, " + owner + "! Let me pull up your latest performance -- " + str(perf.get("views", 0)) + " views, " + str(perf.get("calls", 0)) + " calls this month. Want me to suggest ways to improve?",
            "trial_followup": "Excellent, " + owner + "! I'll prepare the membership upgrade details. Reply CONFIRM to proceed.",
            "regulation_change": "On it, " + owner + "! I'm pulling up the regulatory details now. I'll have a summary of what you need to do ready shortly. Reply CONFIRM to proceed.",
        }

        fallback_body = action_responses.get(kind,
            "On it, " + owner + "! I'm preparing everything based on your current performance (" + str(perf.get("views", 0)) + " views, " + str(perf.get("calls", 0)) + " calls). Reply CONFIRM to proceed.")

        return {
            "action": "send",
            "body": fallback_body,
            "cta": "binary_confirm_cancel",
            "rationale": "Merchant committed to " + kind + "; switching to action mode with context-specific response."
        }

    # --- General reply -- use LLM ---
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
