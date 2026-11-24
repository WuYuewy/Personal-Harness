"""Shared human-readable cost receipts for CLI and UI; no ledger changes or requests."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from personal_harness.costs import CostEntry

if TYPE_CHECKING:
    from personal_harness.runtime import RunResult


def _money(amount: Decimal) -> str:
    # Avoid rounding tiny personal-task costs to zero or printing scientific notation.
    text = format(amount, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _cost_receipt(
    *,
    currency: str,
    cost: Decimal | None,
    reserved_cost: Decimal,
    charges: Sequence[CostEntry],
    requests: int,
    usage_complete: bool,
    reservation_exceeded: bool = False,
    offline: bool = False,
) -> str:
    lines = [f"Cost: {currency} {_money(cost) if cost is not None else 'unknown'}"]
    if cost is None:
        subtotal = sum((entry.amount or Decimal(0) for entry in charges), Decimal(0))
        lines.append(f"Known subtotal: {currency} {_money(subtotal)}")
        lines.append(f"Unknown charges: {sum(entry.amount is None for entry in charges)}")
    if reserved_cost:
        lines.append(f"Budget hold: {currency} {_money(reserved_cost)} (not a billed amount)")
    bases: Counter[str] = Counter(entry.basis for entry in charges)
    labels = {
        "usage": "usage-priced",
        "billed": "bill-verified",
        "estimated": "estimated",
        "unknown": "unknown",
        "offline": "offline",
    }
    if bases:
        lines.append(
            "Cost basis: "
            + ", ".join(
                f"{bases[basis]} {label}" for basis, label in labels.items() if bases[basis]
            )
        )
    elif offline:
        lines.append("Offline rehearsal; no API request sent.")
    elif not requests:
        lines.append("No model request sent.")
    lines.append(f"Attempts: {requests}; usage complete: {'yes' if usage_complete else 'no'}")
    if cost is None:
        lines.append("Check the provider bill for unknown charges before starting another task.")
    if reservation_exceeded:
        lines.append("Recorded cost exceeded its reservation; billing reconciliation is required.")
    return "\n".join(lines)


def run_cost_receipt(result: RunResult) -> str:
    """Describe the immutable end-of-run snapshot, including failed retry charges."""
    receipt = _cost_receipt(
        currency=result.currency,
        cost=result.cost,
        reserved_cost=result.reserved_cost,
        charges=result.charges,
        requests=result.requests,
        usage_complete=result.usage_complete,
        offline=result.provider == "fake",
    )
    if result.history_reads_referenced:
        receipt += (
            f"\nHarness context: {result.history_reads_referenced} old read bodies referenced; "
            f"{result.history_read_bytes_omitted} outgoing bytes omitted across requests "
            "(not tokens or measured money savings)."
        )
    if result.paged_read_results:
        receipt += (
            f"\nHarness read pages: {result.paged_read_results} partial file results; "
            f"{result.read_result_bytes_omitted} body bytes not returned by tools "
            "(sources remain readable; not measured money savings)."
        )
    return (
        receipt
        + _cache_receipt(result.cache_read_tokens, result.cache_savings_cny)
        + _external_receipt(
            len(result.external_charges),
            result.external_cost,
            result.total_cost,
            complete=result.external_cost_complete,
        )
    )


def _external_receipt(
    count: int,
    external_cost: Decimal | str | None,
    total_cost: Decimal | str | None,
    *,
    complete: bool,
) -> str:
    if not count:
        return ""
    amount = f"CNY {external_cost}" if external_cost is not None else "unknown"
    total = f"CNY {total_cost}" if total_cost is not None else "unknown until reconciled"
    status = "" if complete else " External amount requires billing reconciliation."
    return (
        f"\nExternal API charges: {amount} ({count} recorded).{status}"
        f"\nTotal task cost including external APIs: {total}"
    )


def _cache_receipt(tokens: int, savings: Decimal | None) -> str:
    if not tokens:
        return ""
    difference = (
        "unavailable (incomplete usage or price snapshot)"
        if savings is None
        else f"CNY {_money(savings)} vs ordinary input at configured prices; not bill-verified"
    )
    return (
        f"\nProvider-reported cache-read input: {tokens} tokens; "
        f"provider cache price difference: {difference} (not a Personal Harness saving)"
    )


def ledger_cost_receipt(summary: Mapping[str, Any]) -> str:
    """Describe a ledger snapshot without treating missing token usage as free billing."""
    receipt = _cost_receipt(
        currency=summary["currency"],
        cost=Decimal(summary["cost"]) if summary["cost"] is not None else None,
        reserved_cost=Decimal(summary["reserved_cost"]),
        charges=tuple(CostEntry.model_validate(entry) for entry in summary["charges"]),
        requests=summary["requests"],
        usage_complete=summary["usage_complete"],
        reservation_exceeded=summary.get("reservation_exceeded", False),
    )
    savings = summary.get("cache_savings_cny")
    receipt += _cache_receipt(
        summary.get("cache_read_tokens", 0), Decimal(savings) if savings is not None else None
    )
    if summary.get("stop_reason") == "unreconciled_cost":
        receipt += "\nEarlier unresolved charges require billing reconciliation before retrying."
    if summary.get("total_limit_cny") is not None:
        receipt += f"\nLedger total limit for this task: CNY {summary['total_limit_cny']}"
    receipt += _external_receipt(
        len(summary.get("external_charges", ())),
        summary.get("external_cost"),
        summary.get("total_cost"),
        complete=summary.get("external_cost_complete", False),
    )
    if summary.get("stop_reason") == "total_budget":
        receipt += (
            "\nCumulative spending and budget holds leave insufficient room for this request."
        )
    return receipt


def budget_receipt(snapshot: Mapping[str, Any]) -> str:
    """Explain a read-only snapshot; it neither reserves money nor verifies a bill."""
    lines = [f"Profile: {snapshot['profile']}; model: {snapshot['provider']}/{snapshot['model']}"]
    if not snapshot["paid_execution"]:
        return "\n".join(lines + ["Offline provider; no paid request budget applies."])
    lines += [
        f"As of: {snapshot['as_of']}; budget day (UTC): {snapshot['budget_day_utc']}",
        f"One request hold: CNY {_money(Decimal(snapshot['reservation_cny']))} "
        "(not an estimated bill)",
        f"New task limit: CNY {_money(Decimal(snapshot['task_limit_cny']))}",
    ]
    for title, key in (("Daily", "daily"), ("Total", "total")):
        values = snapshot[key]
        remaining = (
            f"CNY {_money(Decimal(values['remaining_cny']))}"
            if values["remaining_cny"] is not None
            else "not capped"
        )
        lines += [
            f"{title} budget remaining: {remaining}; "
            f"exposure: CNY {_money(Decimal(values['exposure_cny']))}",
            f"  Known subtotal: CNY {_money(Decimal(values['known_subtotal_cny']))}; "
            f"hold: CNY {_money(Decimal(values['held_cny']))} (not a billed amount)",
            f"  Requests: {values['usage_priced_requests']} usage-priced, "
            f"{values['bill_verified_requests']} bill-verified; "
            f"{values['inflight_requests']} in flight, {values['unknown_requests']} unknown, "
            f"{values['overrun_requests']} over reservation",
        ]
        if values.get("external_charges"):
            lines.append(f"  External API charges: {values['external_charges']}")
    reasons = {
        "unreconciled_cost": "Earlier unknown/overrun costs require billing reconciliation.",
        "task_budget": "The task limit cannot cover one conservative request hold.",
        "daily_budget": "The daily remaining budget cannot cover one conservative request hold.",
        "total_budget": "The total remaining budget cannot cover one conservative request hold.",
    }
    if snapshot["new_task_request_fits"]:
        lines.append("First request of a new task fits this budget snapshot.")
    else:
        lines.append(f"New task blocked ({snapshot['reason']}): {reasons[snapshot['reason']]}")
    if snapshot["unresolved_runs"]:
        lines.append(
            "Inspect with ledger show RUN_ID --text: " + ", ".join(snapshot["unresolved_runs"])
        )
    lines.append(
        "Read-only snapshot; no money reserved or model request sent. "
        "Execution rechecks atomically; this does not promise a complete task "
        "or provider readiness."
    )
    return "\n".join(lines)
