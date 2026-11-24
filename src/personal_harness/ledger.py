"""Small SQLite request ledger with atomic, Decimal budget admission.

Budget days are UTC request-start dates. Pending reservations survive restarts;
unknown charges require explicit billing reconciliation, never automatic release.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from personal_harness.costs import CostEntry, TokenPrices


class BudgetError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ExternalChargeTracker:
    """Run-scoped external cost API for explicitly authorized SDK tools."""

    def __init__(self, ledger: Ledger, run_id: str) -> None:
        self._ledger = ledger
        self.run_id = run_id

    def reserve(
        self,
        *,
        entry_id: str,
        provider: str,
        maximum_cny: Decimal,
        source: str,
    ) -> CostEntry:
        """Reserve a caller-supplied upper bound before making a paid call."""
        return self._ledger.reserve_external(
            self.run_id,
            entry_id=entry_id,
            provider=provider,
            maximum_cny=maximum_cny,
            source=source,
        )

    def settle(self, charge: CostEntry) -> None:
        """Settle only this run's reservation using measured or reconciled cost."""
        self._ledger.settle_external(charge, run_id=self.run_id)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _budget_reason(
    *,
    unresolved: bool,
    reservation: Decimal,
    task_exposure: Decimal,
    daily_exposure: Decimal,
    total_exposure: Decimal,
    task_limit: Decimal,
    daily_limit: Decimal,
    total_limit: Decimal | None,
) -> str | None:
    """One admission order, also used by a read-only new-task budget snapshot."""
    if unresolved:
        return "unreconciled_cost"
    if task_exposure + reservation > task_limit:
        return "task_budget"
    if daily_exposure + reservation > daily_limit:
        return "daily_budget"
    if total_limit is not None and total_exposure + reservation > total_limit:
        return "total_budget"
    return None


class Ledger:
    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], datetime] = _utc_now,
        read_only: bool = False,
    ) -> None:
        self.path = path
        self.clock = clock
        self.read_only = read_only
        if read_only:
            with self._transaction() as db:
                if db.execute("PRAGMA user_version").fetchone()[0] not in (1, 2):
                    raise ValueError("Unsupported ledger schema version.")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2):
                raise ValueError("Unsupported ledger schema version.")
            db.execute("""CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, started_at TEXT NOT NULL, provider TEXT NOT NULL,
                model TEXT NOT NULL, task_limit TEXT NOT NULL, daily_limit TEXT NOT NULL,
                status TEXT NOT NULL, stop_reason TEXT, metrics_json TEXT NOT NULL DEFAULT '{}'
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                day TEXT NOT NULL, reserved TEXT NOT NULL, state TEXT NOT NULL,
                charge_json TEXT NOT NULL, price_json TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS reconciliations (
                id INTEGER PRIMARY KEY, request_id TEXT NOT NULL REFERENCES requests(id),
                recorded_at TEXT NOT NULL, previous_charge_json TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS run_models (
                run_id TEXT NOT NULL REFERENCES runs(id), provider TEXT NOT NULL,
                model TEXT NOT NULL, PRIMARY KEY(run_id,provider,model)
            )""")
            # External services (for example a future places provider) are
            # recorded beside model requests without pretending they are model
            # attempts. The table is additive so schema 1/2 ledgers remain readable.
            db.execute("""CREATE TABLE IF NOT EXISTS external_charges (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                day TEXT NOT NULL, reserved TEXT NOT NULL, state TEXT NOT NULL,
                charge_json TEXT NOT NULL, recorded_at TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS external_reconciliations (
                id INTEGER PRIMARY KEY, charge_id TEXT NOT NULL REFERENCES external_charges(id),
                recorded_at TEXT NOT NULL, previous_charge_json TEXT NOT NULL
            )""")
            # Additive migration: old runs retain their original unlimited-total policy.
            if version < 2:
                db.execute("ALTER TABLE runs ADD COLUMN total_limit TEXT")
            db.execute("PRAGMA user_version=2")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        db = (
            sqlite3.connect(
                self.path.resolve().as_uri() + "?mode=ro",
                uri=True,
                timeout=10,
                isolation_level=None,
            )
            if self.read_only
            else sqlite3.connect(self.path, timeout=10, isolation_level=None)
        )
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            if self.read_only:
                db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN" if self.read_only else "BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def start_run(
        self,
        run_id: str,
        *,
        provider: str,
        model: str,
        task_limit: Decimal,
        daily_limit: Decimal,
        total_limit: Decimal | None = None,
        fallback_model: tuple[str, str] | None = None,
    ) -> None:
        limits = (task_limit, daily_limit) + ((total_limit,) if total_limit is not None else ())
        if any(not value.is_finite() or value < 0 for value in limits):
            raise ValueError("Budget amounts must be finite and non-negative.")
        with self._transaction() as db:
            db.execute(
                "INSERT INTO runs(id,started_at,provider,model,task_limit,daily_limit,"
                "total_limit,status) VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    self.clock().isoformat(),
                    provider,
                    model,
                    str(task_limit),
                    str(daily_limit),
                    str(total_limit) if total_limit is not None else None,
                    "running",
                ),
            )
            if fallback_model is not None:
                db.execute("INSERT INTO run_models VALUES (?,?,?)", (run_id, *fallback_model))

    @staticmethod
    def _exposure(rows: list[sqlite3.Row]) -> Decimal:
        total = Decimal(0)
        for row in rows:
            if row["state"] in {"reserved", "unknown"}:
                total += Decimal(row["reserved"])
            else:
                amount = CostEntry.model_validate_json(row["charge_json"]).amount or Decimal(0)
                total += (
                    max(amount, Decimal(row["reserved"])) if row["state"] == "estimated" else amount
                )
        return total

    def _check_admission(
        self, db: sqlite3.Connection, run: sqlite3.Row, day: str, amount: Decimal
    ) -> None:
        run_id = run["id"]
        unresolved = db.execute(
            "SELECT 1 FROM requests WHERE state='overrun' OR (state='unknown' AND run_id!=?)",
            (run_id,),
        ).fetchone()
        # Same-run model retries may keep prior holds. Unknown external bills pause all calls.
        if unresolved or self._external_rows(db, states=("unknown", "overrun")):
            raise BudgetError("unreconciled_cost")
        task_rows = db.execute("SELECT * FROM requests WHERE run_id=?", (run_id,)).fetchall()
        task_rows += self._external_rows(db, run_id=run_id)
        daily_rows = db.execute(
            "SELECT * FROM requests WHERE day=? OR state='reserved'", (day,)
        ).fetchall()
        daily_rows += self._external_rows(db, day=day)
        daily_rows += [
            row for row in self._external_rows(db, states=("reserved",)) if row["day"] != day
        ]
        total_limit = Decimal(run["total_limit"]) if run["total_limit"] is not None else None
        all_rows = []
        if total_limit is not None:
            all_rows = db.execute("SELECT * FROM requests").fetchall() + self._external_rows(db)
        reason = _budget_reason(
            unresolved=False,
            reservation=amount,
            task_exposure=self._exposure(task_rows),
            daily_exposure=self._exposure(daily_rows),
            total_exposure=self._exposure(all_rows),
            task_limit=Decimal(run["task_limit"]),
            daily_limit=Decimal(run["daily_limit"]),
            total_limit=total_limit,
        )
        if reason is not None:
            raise BudgetError(reason)

    def reserve(
        self,
        run_id: str,
        charge: CostEntry,
        amount: Decimal,
        *,
        price_json: str,
    ) -> None:
        if charge.category != "model" or charge.basis != "unknown" or charge.currency != "CNY":
            raise ValueError("A new CNY request must start with an unknown charge.")
        if not amount.is_finite() or amount < 0:
            raise ValueError("Reservation must be finite and non-negative.")
        day = self.clock().astimezone(UTC).date().isoformat()
        with self._transaction() as db:
            run = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if run is None or run["status"] not in {"running", "pending_approval"}:
                raise ValueError("Request requires an active run.")
            if (charge.provider, charge.model) != (
                run["provider"],
                run["model"],
            ) and not db.execute(
                "SELECT 1 FROM run_models WHERE run_id=? AND provider=? AND model=?",
                (run_id, charge.provider, charge.model),
            ).fetchone():
                raise ValueError("Request provider/model must match its run.")
            if self._external_rows(db, entry_id=charge.entry_id):
                raise ValueError("Charge entry ID already exists.")
            self._check_admission(db, run, day, amount)
            db.execute(
                "INSERT INTO requests VALUES (?,?,?,?,?,?,?)",
                (
                    charge.entry_id,
                    run_id,
                    day,
                    str(amount),
                    "reserved",
                    charge.model_dump_json(),
                    price_json,
                ),
            )

    def reserve_external(
        self,
        run_id: str,
        *,
        entry_id: str,
        provider: str,
        maximum_cny: Decimal,
        source: str,
    ) -> CostEntry:
        """Atomically reserve an explicit upper bound before an external call."""
        if self.read_only:
            raise ValueError("Cannot reserve an external charge in a read-only ledger.")
        if not maximum_cny.is_finite() or maximum_cny < 0:
            raise ValueError("External reservation must be finite and non-negative.")
        charge = CostEntry(
            entry_id=entry_id,
            category="external_api",
            provider=provider,
            model=None,
            outcome="unknown",
            basis="unknown",
            currency="CNY",
            amount=None,
            source=source,
        )
        day = self.clock().astimezone(UTC).date().isoformat()
        with self._transaction() as db:
            run = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if run is None or run["status"] not in {"running", "pending_approval"}:
                raise ValueError("External reservation requires an active run.")
            duplicate = db.execute(
                "SELECT 1 FROM requests WHERE id=? UNION ALL "
                "SELECT 1 FROM external_charges WHERE id=?",
                (entry_id, entry_id),
            ).fetchone()
            if duplicate:
                raise ValueError("Charge entry ID already exists.")
            self._check_admission(db, run, day, maximum_cny)
            db.execute(
                "INSERT INTO external_charges VALUES (?,?,?,?,?,?,?)",
                (
                    entry_id,
                    run_id,
                    day,
                    str(maximum_cny),
                    "reserved",
                    charge.model_dump_json(),
                    self.clock().isoformat(),
                ),
            )
        return charge

    def settle_external(self, charge: CostEntry, *, run_id: str | None = None) -> None:
        """Settle a previously reserved external call without losing its hold."""
        if self.read_only:
            raise ValueError("Cannot settle an external charge in a read-only ledger.")
        if charge.category != "external_api" or charge.currency != "CNY":
            raise ValueError("External charges must use category external_api and CNY.")
        with self._transaction() as db:
            row = db.execute(
                "SELECT * FROM external_charges WHERE id=?", (charge.entry_id,)
            ).fetchone()
            if row is None or row["state"] != "reserved":
                raise ValueError("External reservation is missing or already settled.")
            if run_id is not None and row["run_id"] != run_id:
                raise ValueError("External reservation belongs to another run.")
            initial = CostEntry.model_validate_json(row["charge_json"])
            if (charge.provider, charge.model, charge.currency, charge.category) != (
                initial.provider,
                initial.model,
                initial.currency,
                initial.category,
            ):
                raise ValueError(
                    "Settlement cannot change the external charge identity or currency."
                )
            if charge.basis not in {"usage", "billed", "estimated", "unknown"}:
                raise ValueError(
                    "External settlement requires usage, billing, estimate or unknown."
                )
            state = "unknown" if charge.amount is None else "settled"
            if (
                charge.amount is not None
                and charge.amount > Decimal(row["reserved"])
                and charge.basis != "billed"
            ):
                state = "overrun"
            elif charge.basis == "estimated":
                state = "estimated"
            if charge.basis == "billed":
                db.execute(
                    "INSERT INTO external_reconciliations("
                    "charge_id,recorded_at,previous_charge_json) VALUES (?,?,?)",
                    (charge.entry_id, self.clock().isoformat(), row["charge_json"]),
                )
            db.execute(
                "UPDATE external_charges SET state=?,charge_json=? WHERE id=?",
                (state, charge.model_dump_json(), charge.entry_id),
            )

    @staticmethod
    def _external_rows(
        db: sqlite3.Connection,
        *,
        run_id: str | None = None,
        entry_id: str | None = None,
        day: str | None = None,
        states: tuple[str, ...] | None = None,
    ) -> list[sqlite3.Row]:
        """Read additive external records, tolerating pre-feature ledgers."""
        clauses: list[str] = []
        values: list[str] = []
        if run_id is not None:
            clauses.append("run_id=?")
            values.append(run_id)
        if entry_id is not None:
            clauses.append("id=?")
            values.append(entry_id)
        if day is not None:
            clauses.append("day=?")
            values.append(day)
        if states:
            clauses.append("state IN (" + ",".join("?" for _ in states) + ")")
            values.extend(states)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        try:
            return db.execute("SELECT * FROM external_charges" + where, values).fetchall()
        except sqlite3.OperationalError as error:
            if "no such table" in str(error).lower():
                return []
            raise

    def record_external_charge(self, run_id: str, charge: CostEntry) -> None:
        """Attach one measured external API charge to an existing run.

        External charges are post-paid records: known amounts affect subsequent
        budget exposure, while an unknown amount blocks further model requests
        until the caller reconciles it. They are kept separate from model
        attempts so token totals and retry counts remain accurate.
        """
        if self.read_only:
            raise ValueError("Cannot record an external charge in a read-only ledger.")
        if charge.category != "external_api" or charge.currency != "CNY":
            raise ValueError("External charges must use category external_api and CNY.")
        if charge.basis not in {"usage", "billed", "estimated", "unknown", "offline"}:
            raise ValueError("Unsupported external charge basis.")
        with self._transaction() as db:
            run = db.execute("SELECT id FROM runs WHERE id=?", (run_id,)).fetchone()
            if run is None:
                raise ValueError("Unknown run ID.")
            duplicate = db.execute(
                "SELECT 1 FROM requests WHERE id=? UNION ALL "
                "SELECT 1 FROM external_charges WHERE id=?",
                (charge.entry_id, charge.entry_id),
            ).fetchone()
            if duplicate is not None:
                raise ValueError("Charge entry ID already exists.")
            state = "unknown" if charge.amount is None else "settled"
            if charge.amount is not None and charge.basis == "estimated":
                state = "estimated"
            db.execute(
                "INSERT INTO external_charges VALUES (?,?,?,?,?,?,?)",
                (
                    charge.entry_id,
                    run_id,
                    self.clock().astimezone(UTC).date().isoformat(),
                    "0",
                    state,
                    charge.model_dump_json(),
                    self.clock().isoformat(),
                ),
            )

    def external_charges(self, run_id: str) -> list[CostEntry]:
        """Return external charges recorded for a run, if supported by its ledger."""
        with self._transaction() as db:
            return [
                CostEntry.model_validate_json(row["charge_json"])
                for row in self._external_rows(db, run_id=run_id)
            ]

    def settle(self, charge: CostEntry) -> None:
        with self._transaction() as db:
            row = db.execute("SELECT * FROM requests WHERE id=?", (charge.entry_id,)).fetchone()
            if row is None or row["state"] not in {"reserved", "unknown", "overrun"}:
                raise ValueError("Request is missing or already settled.")
            initial = CostEntry.model_validate_json(row["charge_json"])
            if (charge.provider, charge.model, charge.currency, charge.category) != (
                initial.provider,
                initial.model,
                initial.currency,
                initial.category,
            ):
                raise ValueError("Settlement cannot change the request identity or currency.")
            if charge.basis not in {"usage", "billed", "unknown"}:
                raise ValueError("Settlement requires usage, billing or an unknown charge.")
            state = "unknown" if charge.amount is None else "settled"
            if (
                charge.amount is not None
                and charge.amount > Decimal(row["reserved"])
                and charge.basis != "billed"
            ):
                state = "overrun"
            if charge.basis == "billed":
                db.execute(
                    "INSERT INTO reconciliations(request_id,recorded_at,previous_charge_json) "
                    "VALUES (?,?,?)",
                    (charge.entry_id, self.clock().isoformat(), row["charge_json"]),
                )
            db.execute(
                "UPDATE requests SET state=?,charge_json=? WHERE id=?",
                (
                    state,
                    charge.model_dump_json(),
                    charge.entry_id,
                ),
            )

    def reconcile(self, request_id: str, amount: Decimal, *, source: str) -> None:
        with self._transaction() as db:
            row = db.execute(
                "SELECT charge_json FROM requests WHERE id=?", (request_id,)
            ).fetchone()
            if row is not None:
                data = json.loads(row["charge_json"])
            else:
                external = db.execute(
                    "SELECT state,charge_json FROM external_charges WHERE id=?", (request_id,)
                ).fetchone()
                if external is None:
                    raise ValueError("Unknown request ID.")
                data = json.loads(external["charge_json"])
                if external["state"] not in {
                    "reserved",
                    "unknown",
                    "overrun",
                    "estimated",
                } and not (
                    data["basis"] == "estimated"  # Pre-existing post-paid rows used settled.
                ):
                    raise ValueError("External charge is not awaiting reconciliation.")
                replacement = CostEntry.model_validate(
                    {**data, "basis": "billed", "amount": amount, "source": source}
                )
                db.execute(
                    "INSERT INTO external_reconciliations("
                    "charge_id,recorded_at,previous_charge_json) "
                    "VALUES (?,?,?)",
                    (request_id, self.clock().isoformat(), external["charge_json"]),
                )
                db.execute(
                    "UPDATE external_charges SET state='settled',charge_json=? WHERE id=?",
                    (replacement.model_dump_json(), request_id),
                )
                return
        self.settle(
            CostEntry.model_validate(
                {**data, "basis": "billed", "amount": amount, "source": source}
            )
        )

    def finish_run(
        self,
        run_id: str,
        status: str,
        *,
        reason: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        with self._transaction() as db:
            db.execute(
                "UPDATE runs SET status=?,stop_reason=?,metrics_json=? WHERE id=?",
                (
                    status,
                    reason,
                    json.dumps(metrics or {}),
                    run_id,
                ),
            )
            db.execute(
                "UPDATE external_charges SET state='unknown' WHERE run_id=? AND state='reserved'",
                (run_id,),
            )

    def summary(self, run_id: str) -> dict[str, Any]:
        with self._transaction() as db:
            run = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if run is None:
                raise ValueError("Unknown run ID.")
            rows = db.execute(
                "SELECT * FROM requests WHERE run_id=? ORDER BY rowid", (run_id,)
            ).fetchall()
            external_rows = self._external_rows(db, run_id=run_id)
        charges = [CostEntry.model_validate_json(row["charge_json"]) for row in rows]
        external_charges = [
            CostEntry.model_validate_json(row["charge_json"]) for row in external_rows
        ]
        cache_read_tokens = 0
        cache_savings = Decimal(0)
        cache_savings_complete = True
        for row, charge in zip(rows, charges, strict=True):
            if charge.usage is None:
                cache_savings_complete = False
                continue
            if charge.usage.cache_read_tokens == 0:
                continue
            cache_read_tokens += charge.usage.cache_read_tokens
            if charge.basis != "usage":
                cache_savings_complete = False
                continue
            try:
                prices = TokenPrices.model_validate_json(row["price_json"])
            except ValidationError:
                # Old/imported records may lack a complete price snapshot.
                cache_savings_complete = False
                continue
            if (prices.provider, prices.model, prices.currency) != (
                charge.provider,
                charge.model,
                charge.currency,
            ):
                cache_savings_complete = False
                continue
            if prices.cache_read_per_million is None:
                cache_savings_complete = False
                continue
            cache_savings += (
                charge.usage.cache_read_tokens
                * max(Decimal(0), prices.input_per_million - prices.cache_read_per_million)
                / Decimal(1_000_000)
            )
        verified = sum((entry.amount or Decimal(0) for entry in charges), Decimal(0))
        held = sum(
            (Decimal(row["reserved"]) for row in rows if row["state"] in {"reserved", "unknown"}),
            Decimal(0),
        )
        external_held = sum(
            (
                Decimal(row["reserved"])
                for row in external_rows
                if row["state"] in {"reserved", "unknown", "estimated"}
            ),
            Decimal(0),
        )
        complete = all(entry.amount is not None for entry in charges)
        usage_complete = all(entry.usage is not None for entry in charges)
        external_subtotal = sum(
            (entry.amount or Decimal(0) for entry in external_charges), Decimal(0)
        )
        external_verified = sum(
            (
                entry.amount or Decimal(0)
                for entry in external_charges
                if entry.basis != "estimated"
            ),
            Decimal(0),
        )
        external_state = {row["id"]: row["state"] for row in external_rows}
        external_amount_complete = all(entry.amount is not None for entry in external_charges)
        external_complete = external_amount_complete and all(
            entry.basis != "estimated" and external_state.get(entry.entry_id) != "overrun"
            for entry in external_charges
        )
        return dict(
            run_id=run_id,
            status=run["status"],
            stop_reason=run["stop_reason"],
            currency="CNY",
            cost=str(verified) if complete else None,
            verified_subtotal=str(verified),
            reserved_cost=str(held + external_held),
            cost_complete=complete,
            usage_complete=usage_complete,
            requests=len(charges),
            charges=[entry.model_dump(mode="json") for entry in charges],
            external_charges=[entry.model_dump(mode="json") for entry in external_charges],
            external_verified_subtotal=str(external_verified),
            external_cost=(str(external_subtotal) if external_amount_complete else None),
            external_cost_complete=external_complete,
            total_cost=(
                str(verified + external_subtotal) if complete and external_complete else None
            ),
            total_cost_complete=complete and external_complete,
            cache_read_tokens=cache_read_tokens,
            cache_savings_cny=str(cache_savings) if cache_savings_complete else None,
            cache_savings_complete=cache_savings_complete,
            metrics=json.loads(run["metrics_json"]),
            reservation_exceeded=any(row["state"] == "overrun" for row in (*rows, *external_rows)),
            total_limit_cny=run["total_limit"] if "total_limit" in run.keys() else None,
        )

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not 1 <= limit <= 100:
            raise ValueError("Limit must be between 1 and 100.")
        with self._transaction() as db:
            ids = [
                row[0]
                for row in db.execute("SELECT id FROM runs ORDER BY rowid DESC LIMIT ?", (limit,))
            ]
        return [self.summary(run_id) for run_id in ids]

    def total_exposure(self) -> Decimal:
        """All recorded spending plus unresolved holds, without a daily reset."""
        with self._transaction() as db:
            return self._exposure(db.execute("SELECT * FROM requests").fetchall()) + self._exposure(
                self._external_rows(db)
            )
