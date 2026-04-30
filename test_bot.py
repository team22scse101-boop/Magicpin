"""Quick smoke test for the Vera bot."""
import json
import sys
import io
from urllib import request as urlrequest
from pathlib import Path

# Fix Windows encoding for emoji
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BOT = "http://localhost:8080"
DATASET = Path("dataset")

def post(path, body):
    data = json.dumps(body).encode("utf-8")
    req = urlrequest.Request(f"{BOT}{path}", data=data,
                             headers={"Content-Type": "application/json"})
    resp = urlrequest.urlopen(req, timeout=30)
    return json.loads(resp.read().decode("utf-8"))

def get(path):
    req = urlrequest.Request(f"{BOT}{path}")
    resp = urlrequest.urlopen(req, timeout=10)
    return json.loads(resp.read().decode("utf-8"))

def main():
    # 1. Healthz
    print("=== HEALTHZ ===")
    r = get("/v1/healthz")
    print(json.dumps(r, indent=2))

    # 2. Metadata
    print("\n=== METADATA ===")
    r = get("/v1/metadata")
    print(f"Team: {r.get('team_name')}, Model: {r.get('model')}")

    # 3. Push categories
    print("\n=== PUSHING CATEGORIES ===")
    for f in (DATASET / "categories").glob("*.json"):
        cat = json.load(open(f))
        r = post("/v1/context", {"scope": "category", "context_id": cat["slug"],
                                  "version": 1, "payload": cat,
                                  "delivered_at": "2026-04-29T10:00:00Z"})
        print(f"  {cat['slug']}: accepted={r.get('accepted')}")

    # 4. Push seed merchants
    print("\n=== PUSHING MERCHANTS ===")
    merchants = json.load(open(DATASET / "merchants_seed.json"))["merchants"]
    for m in merchants:
        r = post("/v1/context", {"scope": "merchant", "context_id": m["merchant_id"],
                                  "version": 1, "payload": m,
                                  "delivered_at": "2026-04-29T10:00:00Z"})
        print(f"  {m['merchant_id'][:30]}: accepted={r.get('accepted')}")

    # 5. Push seed triggers
    print("\n=== PUSHING TRIGGERS ===")
    triggers = json.load(open(DATASET / "triggers_seed.json"))["triggers"]
    for t in triggers:
        r = post("/v1/context", {"scope": "trigger", "context_id": t["id"],
                                  "version": 1, "payload": t,
                                  "delivered_at": "2026-04-29T10:00:00Z"})
        print(f"  {t['id'][:40]}: accepted={r.get('accepted')}")

    # 6. Healthz after push
    print("\n=== HEALTHZ AFTER PUSH ===")
    r = get("/v1/healthz")
    print(json.dumps(r, indent=2))

    # 7. Tick with first 3 triggers
    print("\n=== TICK TEST (3 triggers) ===")
    tids = [t["id"] for t in triggers[:3]]
    r = post("/v1/tick", {"now": "2026-04-29T10:35:00Z", "available_triggers": tids})
    actions = r.get("actions", [])
    print(f"  Actions returned: {len(actions)}")
    for a in actions:
        print(f"\n  --- Action ---")
        print(f"  Merchant: {a.get('merchant_id', '?')[:30]}")
        print(f"  Trigger:  {a.get('trigger_id', '?')[:40]}")
        print(f"  Send as:  {a.get('send_as')}")
        print(f"  CTA:      {a.get('cta')}")
        print(f"  Body:     {a.get('body', '')[:200]}")
        print(f"  Rationale: {a.get('rationale', '')[:150]}")

    # 8. Reply test — auto-reply detection
    print("\n=== AUTO-REPLY TEST ===")
    r = post("/v1/reply", {
        "conversation_id": "conv_test_auto",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "customer_id": None,
        "from_role": "merchant",
        "message": "Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly.",
        "received_at": "2026-04-29T10:42:00Z",
        "turn_number": 2
    })
    print(f"  Action: {r.get('action')}")
    print(f"  Body: {r.get('body', '')[:150]}")
    print(f"  Rationale: {r.get('rationale', '')[:150]}")

    # 9. Reply test — hostile
    print("\n=== HOSTILE TEST ===")
    r = post("/v1/reply", {
        "conversation_id": "conv_test_hostile",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "from_role": "merchant",
        "message": "Stop messaging me. This is useless spam.",
        "received_at": "2026-04-29T10:42:00Z",
        "turn_number": 2
    })
    print(f"  Action: {r.get('action')}")
    print(f"  Body: {r.get('body', '')[:150]}")

    # 10. Reply test — intent commitment
    print("\n=== INTENT TRANSITION TEST ===")
    r = post("/v1/reply", {
        "conversation_id": "conv_test_intent",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "from_role": "merchant",
        "message": "Ok lets do it. Whats next?",
        "received_at": "2026-04-29T10:42:00Z",
        "turn_number": 2
    })
    print(f"  Action: {r.get('action')}")
    print(f"  Body: {r.get('body', '')[:200]}")
    print(f"  CTA: {r.get('cta')}")

    print("\n=== ALL TESTS COMPLETE ===")

if __name__ == "__main__":
    main()
