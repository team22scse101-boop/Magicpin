"""
Versioned, in-memory context store.

Thread-safe storage for category, merchant, customer, and trigger contexts.
Supports idempotent upsert with version conflict detection.
"""
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


class ContextStore:
    """Thread-safe versioned context store."""

    def __init__(self):
        # (scope, context_id) -> {"version": int, "payload": dict, "stored_at": str}
        self._store: Dict[Tuple[str, str], dict] = {}
        self._lock = threading.Lock()
        # conversation state: conversation_id -> list of turns
        self._conversations: Dict[str, list] = {}
        # suppression keys: key -> expiry timestamp
        self._suppressed: Dict[str, float] = {}
        # sent message hashes per conversation for anti-repetition
        self._sent_hashes: Dict[str, set] = {}

    # ─── Context CRUD ─────────────────────────────────────────────────────

    def upsert(self, scope: str, context_id: str, version: int,
               payload: dict) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Upsert a context. Returns (accepted, reason, current_version).
        - If no existing entry or version > existing → accept & store.
        - If version <= existing → reject with stale_version.
        """
        with self._lock:
            key = (scope, context_id)
            existing = self._store.get(key)

            if existing and existing["version"] >= version:
                return False, "stale_version", existing["version"]

            self._store[key] = {
                "version": version,
                "payload": payload,
                "stored_at": datetime.now(timezone.utc).isoformat(),
            }
            return True, None, version

    def get(self, scope: str, context_id: str) -> Optional[dict]:
        """Get payload for a context."""
        with self._lock:
            entry = self._store.get((scope, context_id))
            return entry["payload"] if entry else None

    def get_version(self, scope: str, context_id: str) -> Optional[int]:
        """Get current version for a context."""
        with self._lock:
            entry = self._store.get((scope, context_id))
            return entry["version"] if entry else None

    def count_by_scope(self) -> Dict[str, int]:
        """Count contexts by scope."""
        with self._lock:
            counts: Dict[str, int] = {}
            for (scope, _), _ in self._store.items():
                counts[scope] = counts.get(scope, 0) + 1
            return counts

    # ─── Context Lookup Helpers ───────────────────────────────────────────

    def get_trigger(self, trigger_id: str) -> Optional[dict]:
        """Get a trigger context by ID."""
        return self.get("trigger", trigger_id)

    def get_merchant(self, merchant_id: str) -> Optional[dict]:
        """Get a merchant context by ID."""
        return self.get("merchant", merchant_id)

    def get_customer(self, customer_id: str) -> Optional[dict]:
        """Get a customer context by ID."""
        return self.get("customer", customer_id)

    def get_category(self, slug: str) -> Optional[dict]:
        """Get a category context by slug."""
        return self.get("category", slug)

    def get_category_for_merchant(self, merchant: dict) -> Optional[dict]:
        """Get the category context for a merchant."""
        slug = merchant.get("category_slug", "")
        return self.get_category(slug)

    # ─── Conversation State ───────────────────────────────────────────────

    def add_conversation_turn(self, conversation_id: str, turn: dict):
        """Record a conversation turn."""
        with self._lock:
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = []
            self._conversations[conversation_id].append(turn)

    def get_conversation(self, conversation_id: str) -> List[dict]:
        """Get all turns for a conversation."""
        with self._lock:
            return list(self._conversations.get(conversation_id, []))

    def get_conversation_turn_count(self, conversation_id: str) -> int:
        """Get the number of turns in a conversation."""
        with self._lock:
            return len(self._conversations.get(conversation_id, []))

    # ─── Suppression ──────────────────────────────────────────────────────

    def is_suppressed(self, key: str) -> bool:
        """Check if a suppression key is active."""
        with self._lock:
            expiry = self._suppressed.get(key)
            if expiry is None:
                return False
            now = datetime.now(timezone.utc).timestamp()
            if now > expiry:
                del self._suppressed[key]
                return False
            return True

    def suppress(self, key: str, ttl_seconds: int = 86400):
        """Mark a suppression key as active."""
        with self._lock:
            now = datetime.now(timezone.utc).timestamp()
            self._suppressed[key] = now + ttl_seconds

    # ─── Anti-repetition ──────────────────────────────────────────────────

    def is_repeated(self, conversation_id: str, body: str) -> bool:
        """Check if this exact body was already sent in this conversation."""
        body_hash = hash(body.strip().lower())
        with self._lock:
            sent = self._sent_hashes.get(conversation_id, set())
            return body_hash in sent

    def record_sent(self, conversation_id: str, body: str):
        """Record a sent message hash."""
        body_hash = hash(body.strip().lower())
        with self._lock:
            if conversation_id not in self._sent_hashes:
                self._sent_hashes[conversation_id] = set()
            self._sent_hashes[conversation_id].add(body_hash)

    # ─── Ended conversations ──────────────────────────────────────────────

    def is_conversation_ended(self, conversation_id: str) -> bool:
        """Check if a conversation has been ended."""
        with self._lock:
            turns = self._conversations.get(conversation_id, [])
            return any(t.get("action") == "end" for t in turns)

    def end_conversation(self, conversation_id: str, rationale: str = ""):
        """Mark a conversation as ended."""
        self.add_conversation_turn(conversation_id, {
            "action": "end",
            "rationale": rationale,
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    # ─── Active merchant conversations ────────────────────────────────────

    def get_active_conversation_for_merchant(self, merchant_id: str) -> Optional[str]:
        """Get active conversation ID for a merchant (most recent non-ended)."""
        with self._lock:
            for conv_id, turns in self._conversations.items():
                if any(t.get("merchant_id") == merchant_id for t in turns):
                    if not any(t.get("action") == "end" for t in turns):
                        return conv_id
            return None

    # ─── Merchant suppression (hostile exit) ──────────────────────────────

    def suppress_merchant(self, merchant_id: str, days: int = 30):
        """Suppress all triggers for a merchant."""
        self.suppress(f"merchant_suppressed:{merchant_id}", days * 86400)

    def is_merchant_suppressed(self, merchant_id: str) -> bool:
        """Check if a merchant is suppressed."""
        return self.is_suppressed(f"merchant_suppressed:{merchant_id}")
