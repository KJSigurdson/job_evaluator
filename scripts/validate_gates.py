"""One-off audit: validate every user's stored gate config (scoring_weights.
location_rule / seniority_rule) against a strict schema (LocationRule/
SeniorityRule — unknown keys rejected, seniority levels constrained to the known
five buckets). Read-only, never writes to Supabase.

This is a STRICTER check than the live pipeline's user_store._parse_rule(), which
only requires syntactically-valid JSON that parses to an object — a row can pass
_parse_rule() (so the live pipeline still processes that user normally) while
failing this stricter validator (e.g. an unknown key, or a seniority level outside
the known five buckets). That's worth surfacing here even though it doesn't
exclude the user from a live run today; this script's job is to make the current
state of every user's stored config visible, not to change pipeline behavior.

Usage:
    python -m scripts.validate_gates
"""
from __future__ import annotations

import json

from pydantic import ValidationError

from src.schemas import LocationRule, SeniorityRule
from src.supabase_client import get_client

# load_dotenv() is deliberately NOT called at module import time (unlike this
# repo's other scripts/smoke_llm.py-style entry points) — this module is imported
# directly by tests/test_validate_gates.py to unit-test _validate_field(), and a
# module-level load_dotenv() would leak real credentials into os.environ for the
# rest of the pytest process (os.environ mutations aren't test-scoped), letting
# later tests silently make real Anthropic/Supabase calls instead of using their
# stubs. Called only in the __main__ guard below, which import never executes.


def _validate_field(raw_value, model: type) -> tuple[bool, str]:
    """Parse *raw_value* (mirroring user_store._parse_rule's None/dict/str handling)
    then validate it against *model*. Returns (is_valid, detail)."""
    if raw_value is None:
        return True, "(unset — permissive default)"
    if isinstance(raw_value, dict):
        parsed = raw_value
    elif isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return True, "(unset — permissive default)"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON: {exc}"
    else:
        return False, f"unexpected type {type(raw_value).__name__}"

    try:
        model.model_validate(parsed)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in exc.errors()
        )
        return False, details
    return True, "ok"


def main() -> None:
    client = get_client()
    rows = client.table("scoring_weights").select("user_id,location_rule,seniority_rule").execute().data or []
    rows.sort(key=lambda r: r["user_id"])

    print(f"{'user_id':40} {'status':9} detail")
    print("-" * 110)

    invalid_count = 0
    for row in rows:
        uid = row["user_id"]
        loc_ok, loc_detail = _validate_field(row.get("location_rule"), LocationRule)
        sen_ok, sen_detail = _validate_field(row.get("seniority_rule"), SeniorityRule)
        ok = loc_ok and sen_ok
        if not ok:
            invalid_count += 1

        parts = []
        if not loc_ok:
            parts.append(f"location_rule: {loc_detail}")
        if not sen_ok:
            parts.append(f"seniority_rule: {sen_detail}")
        detail = "; ".join(parts) if parts else "ok"

        print(f"{uid:40} {'valid' if ok else 'INVALID':9} {detail}")

    print("-" * 110)
    print(f"{len(rows)} user(s) checked — {invalid_count} invalid, {len(rows) - invalid_count} valid")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
