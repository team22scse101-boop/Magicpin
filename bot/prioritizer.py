"""
Signal prioritizer — ranks triggers for a given tick.
Picks the single best trigger per merchant to maximize scoring.
"""
from typing import Dict, List, Optional
from bot.context_store import ContextStore


def prioritize_triggers(store: ContextStore, trigger_ids: List[str],
                        now: str) -> List[dict]:
    """
    Rank triggers by priority and return ordered list of actionable triggers.
    Filters out suppressed, expired, and merchant-suppressed triggers.
    Returns list of dicts: {trigger_id, trigger, merchant, category, customer, score}
    """
    scored: List[dict] = []
    seen_merchants: set = set()

    for tid in trigger_ids:
        trigger = store.get_trigger(tid)
        if not trigger:
            continue

        merchant_id = trigger.get("merchant_id")
        if not merchant_id:
            continue

        # Skip if merchant is globally suppressed (hostile exit)
        if store.is_merchant_suppressed(merchant_id):
            continue

        # Skip if this specific trigger's suppression key is active
        supp_key = trigger.get("suppression_key", "")
        if supp_key and store.is_suppressed(supp_key):
            continue

        # Skip if we already have an action for this merchant this tick
        if merchant_id in seen_merchants:
            continue

        merchant = store.get_merchant(merchant_id)
        if not merchant:
            continue

        category = store.get_category_for_merchant(merchant)
        if not category:
            continue

        customer = None
        customer_id = trigger.get("customer_id")
        if customer_id:
            customer = store.get_customer(customer_id)

        # Score this trigger
        score = _score_trigger(trigger, merchant, store)

        scored.append({
            "trigger_id": tid,
            "trigger": trigger,
            "merchant": merchant,
            "merchant_id": merchant_id,
            "category": category,
            "customer": customer,
            "customer_id": customer_id,
            "score": score,
        })
        seen_merchants.add(merchant_id)

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:20]  # Cap at 20 actions per tick


def _score_trigger(trigger: dict, merchant: dict, store: ContextStore) -> float:
    """Score a trigger for prioritization."""
    score = 0.0

    # Base urgency (0-10)
    urgency = trigger.get("urgency", 1)
    score += urgency * 2

    # Engagement recency bonus
    signals = merchant.get("signals", [])
    if any("engaged_in_last_24h" in s for s in signals):
        score += 3
    elif any("engaged_in_last_48h" in s for s in signals):
        score += 2
    elif any("dormant" in s for s in signals):
        score -= 1

    # High-urgency trigger kinds
    high_urgency_kinds = {"supply_alert", "regulation_change", "recall_due",
                          "chronic_refill_due", "active_planning_intent"}
    if trigger.get("kind") in high_urgency_kinds:
        score += 3

    # Customer-scoped triggers with consent
    if trigger.get("scope") == "customer" and trigger.get("customer_id"):
        customer = store.get_customer(trigger["customer_id"])
        if customer and customer.get("consent", {}).get("scope"):
            score += 1

    # Active subscription bonus
    sub = merchant.get("subscription", {})
    if sub.get("status") == "active":
        score += 1
    elif sub.get("status") == "trial":
        score += 0.5

    return score
