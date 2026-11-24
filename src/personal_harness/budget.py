"""Read-only budget snapshot for the first request of a new personal task."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from personal_harness.config import ResolvedConfig
from personal_harness.connections import model_prices
from personal_harness.costs import CostEntry, request_reservation
from personal_harness.ledger import Ledger, _budget_reason


def _request_snapshot(path: Path) -> tuple[bool, list[sqlite3.Row], list[sqlite3.Row]]:
    # Avoid Ledger construction: it creates directories and migrates old schemas.
    if not path.exists():
        return False, [], []
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        if db.execute("PRAGMA user_version").fetchone()[0] not in (1, 2):
            raise ValueError("Unsupported ledger schema version.")
        rows = db.execute("SELECT * FROM requests ORDER BY rowid").fetchall()
        # Refuse malformed accounting instead of displaying an apparently usable balance.
        for row in rows:
            charge = CostEntry.model_validate_json(row["charge_json"])
            reserved = Decimal(row["reserved"])
            held = row["state"] in {"reserved", "unknown"}
            if (
                row["state"] not in {"reserved", "unknown", "settled", "overrun"}
                or not reserved.is_finite()
                or reserved < 0
                or charge.currency != "CNY"
                or charge.category != "model"
                or charge.basis not in {"unknown", "usage", "billed"}
                or charge.entry_id != row["id"]
                or (charge.amount is None) != held
                or date.fromisoformat(row["day"]).isoformat() != row["day"]
            ):
                raise ValueError("Invalid ledger request record.")
        try:
            external_rows = db.execute("SELECT * FROM external_charges ORDER BY rowid").fetchall()
        except sqlite3.OperationalError as error:
            if "no such table" not in str(error).lower():
                raise
            external_rows = []
        for row in external_rows:
            charge = CostEntry.model_validate_json(row["charge_json"])
            reserved = Decimal(row["reserved"])
            held = row["state"] in {"reserved", "unknown"}
            if (
                row["state"] not in {"reserved", "unknown", "settled", "estimated", "overrun"}
                or not reserved.is_finite()
                or reserved < 0
                or charge.category != "external_api"
                or charge.currency != "CNY"
                or charge.entry_id != row["id"]
                or (charge.amount is None) != held
                or (held and charge.basis != "unknown")
                or (row["state"] == "estimated" and charge.basis != "estimated")
                or (
                    row["state"] == "overrun"
                    and (
                        charge.basis not in {"estimated", "usage"}
                        or charge.amount is None
                        or charge.amount <= reserved
                    )
                )
                or date.fromisoformat(row["day"]).isoformat() != row["day"]
            ):
                raise ValueError("Invalid external charge record.")
        return True, rows, external_rows


def budget_statuses(
    resolved_configs: Sequence[ResolvedConfig], *, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Inspect profiles against one consistent, read-only ledger snapshot."""
    if not resolved_configs:
        raise ValueError("At least one profile is required for budget inspection.")
    paths = {
        resolved.source_file.parent / ".personal-harness/ledger.sqlite3"
        for resolved in resolved_configs
    }
    if len(paths) != 1:
        raise ValueError("All profiles must share the same configuration directory.")
    current = now if now is not None else datetime.now(UTC)
    if current.utcoffset() is None:
        raise ValueError("Budget snapshot time must include a timezone.")
    if any(resolved.config.model.provider != "fake" for resolved in resolved_configs):
        path = next(iter(paths))
        try:
            exists, rows, external_rows = _request_snapshot(path)
        except (OSError, sqlite3.Error, ValueError, InvalidOperation, IndexError, TypeError):
            raise ValueError(
                "Cannot inspect budget ledger; check its format, schema and availability."
            ) from None
    else:
        exists, rows, external_rows = False, [], []
    return [
        _status_from_snapshot(
            resolved,
            current,
            exists=exists,
            rows=rows,
            external_rows=external_rows,
        )
        for resolved in resolved_configs
    ]


def budget_status(resolved: ResolvedConfig, *, now: datetime | None = None) -> dict[str, Any]:
    """Inspect current limits and ledger without keys, network, migration or reservations."""
    return budget_statuses([resolved], now=now)[0]


def _status_from_snapshot(
    resolved: ResolvedConfig,
    current: datetime,
    *,
    exists: bool,
    rows: list[sqlite3.Row],
    external_rows: list[sqlite3.Row],
) -> dict[str, Any]:
    config = resolved.config
    common: dict[str, Any] = {
        "schema_version": 1,
        "as_of": current.astimezone(UTC).isoformat(),
        "profile": resolved.profile,
        "provider": config.model.provider,
        "model": config.model.model,
        "scope": "first_request_of_new_task",
        "read_only": True,
    }
    if config.model.provider == "fake":
        return {**common, "paid_execution": False, "reason": "offline_provider"}
    prices, context_window = model_prices(config.model)
    reservation = request_reservation(prices, context_window, config.model.max_output_tokens)
    day = current.astimezone(UTC).date().isoformat()
    all_rows = [*rows, *external_rows]
    daily_rows = [row for row in all_rows if row["day"] == day or row["state"] == "reserved"]
    blocked = [row for row in all_rows if row["state"] in {"unknown", "overrun"}]
    daily_exposure, total_exposure = Ledger._exposure(daily_rows), Ledger._exposure(all_rows)
    limits = config.budget
    assert limits.max_task_cny is not None and limits.max_daily_cny is not None
    reason = _budget_reason(
        unresolved=bool(blocked),
        reservation=reservation,
        task_exposure=Decimal(0),
        daily_exposure=daily_exposure,
        total_exposure=total_exposure,
        task_limit=limits.max_task_cny,
        daily_limit=limits.max_daily_cny,
        total_limit=limits.max_total_cny,
    )

    def amounts(selected: list[sqlite3.Row], limit: Decimal | None) -> dict[str, Any]:
        entries = [CostEntry.model_validate_json(row["charge_json"]) for row in selected]
        exposure = Ledger._exposure(selected)
        known = sum(
            (entry.amount or Decimal(0) for entry in entries if entry.basis != "estimated"),
            Decimal(0),
        )
        return {
            "limit_cny": str(limit) if limit is not None else None,
            "exposure_cny": str(exposure),
            "remaining_cny": str(limit - exposure) if limit is not None else None,
            "known_subtotal_cny": str(known),
            "held_cny": str(exposure - known),
            "inflight_requests": sum(row["state"] == "reserved" for row in selected),
            "unknown_requests": sum(row["state"] == "unknown" for row in selected),
            "overrun_requests": sum(row["state"] == "overrun" for row in selected),
            "usage_priced_requests": sum(
                e.category == "model" and e.basis == "usage" for e in entries
            ),
            "bill_verified_requests": sum(
                e.category == "model" and e.basis == "billed" for e in entries
            ),
            "external_charges": sum(e.category == "external_api" for e in entries),
        }

    return {
        **common,
        "paid_execution": True,
        "currency": "CNY",
        "ledger_exists": exists,
        "budget_day_utc": day,
        "reservation_cny": str(reservation),
        "fallback_reservation_cny": (
            str(
                request_reservation(
                    *model_prices(config.fallback_model), config.fallback_model.max_output_tokens
                )
            )
            if config.fallback_model is not None
            else None
        ),
        "task_limit_cny": str(limits.max_task_cny),
        "daily": amounts(daily_rows, limits.max_daily_cny),
        "total": amounts(all_rows, limits.max_total_cny),
        "new_task_request_fits": reason is None,
        "reason": reason,
        "unresolved_runs": list(dict.fromkeys(row["run_id"] for row in blocked)),
    }
