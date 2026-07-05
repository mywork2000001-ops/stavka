"""Paper trading journal: track model predictions against ACTUAL outcomes and
closing odds, without any real stakes.

This is the validation step bukmeker.txt §3.7 calls for before ever betting
real money, and the one still missing after v11's backtest: a backtest only
proves a model wasn't obviously wrong on OLD data it was fit on (minus the
holdout); paper trading proves -- or disproves -- it on new matches the model
has never touched, over the weeks/months it actually takes for variance to
average out. It's also how CLV (Closing Line Value) gets measured at all:
comparing the odds taken at bet-placement time against the closing odds just
before kickoff is the earliest reliable signal of real edge, well before
enough settled bets have accumulated for win/loss alone to mean anything
(bukmeker.txt §1.4's own CLV definition: `log(closing / opening)`).

Persisted to a JSON file (not `st.session_state`) on purpose: unlike the
dashboard's session-only coupon history, a paper-trading journal is only
useful if it survives across many dashboard restarts over real time.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PaperBet:
    bet_id: int
    match_label: str
    outcome_label: str
    model_prob: float
    odds_taken: float
    stake: float
    placed_at: datetime
    closing_odds: float | None = None
    actual_outcome: bool | None = None  # True = won, False = lost, None = not yet settled
    settled_at: datetime | None = None

    @property
    def is_settled(self) -> bool:
        return self.actual_outcome is not None

    @property
    def profit(self) -> float:
        if not self.is_settled:
            raise ValueError(f"bet {self.bet_id} is not settled yet -- profit is undefined until it is")
        return self.stake * (self.odds_taken - 1.0) if self.actual_outcome else -self.stake

    @property
    def clv(self) -> float | None:
        """log(closing_odds / odds_taken); positive means you beat the closing
        line (got a better price than the market settled on) -- `None` until
        a closing odds has been recorded."""
        if self.closing_odds is None:
            return None
        return math.log(self.closing_odds / self.odds_taken)


@dataclass
class PaperTradingJournal:
    bets: list[PaperBet] = field(default_factory=list)

    def log_bet(
        self,
        match_label: str,
        outcome_label: str,
        model_prob: float,
        odds_taken: float,
        stake: float,
        placed_at: datetime | None = None,
    ) -> PaperBet:
        if odds_taken <= 1.0:
            raise ValueError(f"odds_taken must be > 1.0, got {odds_taken}")
        if not (0.0 <= model_prob <= 1.0):
            raise ValueError(f"model_prob must be in [0, 1], got {model_prob}")
        bet_id = 1 if not self.bets else max(b.bet_id for b in self.bets) + 1
        bet = PaperBet(
            bet_id=bet_id,
            match_label=match_label,
            outcome_label=outcome_label,
            model_prob=model_prob,
            odds_taken=odds_taken,
            stake=stake,
            placed_at=placed_at or datetime.now(),
        )
        self.bets.append(bet)
        return bet

    def settle_bet(
        self,
        bet_id: int,
        won: bool,
        closing_odds: float | None = None,
        settled_at: datetime | None = None,
    ) -> PaperBet:
        bet = next((b for b in self.bets if b.bet_id == bet_id), None)
        if bet is None:
            raise ValueError(f"no bet with bet_id={bet_id}")
        bet.actual_outcome = won
        bet.closing_odds = closing_odds
        bet.settled_at = settled_at or datetime.now()
        return bet

    def settled_bets(self) -> list[PaperBet]:
        return [b for b in self.bets if b.is_settled]

    def pending_bets(self) -> list[PaperBet]:
        return [b for b in self.bets if not b.is_settled]

    def summary(self) -> dict:
        """Aggregate stats over SETTLED bets only: count, hit rate, total
        staked, total profit, ROI, and average CLV (over the subset with a
        recorded closing odds). All `None` until at least one bet is settled
        -- there is nothing honest to report yet, not a zero."""
        settled = self.settled_bets()
        if not settled:
            return {
                "n_settled": 0,
                "n_pending": len(self.pending_bets()),
                "hit_rate": None,
                "total_staked": 0.0,
                "total_profit": 0.0,
                "roi": None,
                "avg_clv": None,
            }

        total_staked = sum(b.stake for b in settled)
        total_profit = sum(b.profit for b in settled)
        wins = sum(1 for b in settled if b.actual_outcome)
        clvs = [b.clv for b in settled if b.clv is not None]
        return {
            "n_settled": len(settled),
            "n_pending": len(self.pending_bets()),
            "hit_rate": wins / len(settled),
            "total_staked": total_staked,
            "total_profit": total_profit,
            "roi": total_profit / total_staked if total_staked > 0 else None,
            "avg_clv": sum(clvs) / len(clvs) if clvs else None,
        }


def save_journal(journal: PaperTradingJournal, path: str) -> None:
    data = {
        "bets": [
            {
                "bet_id": b.bet_id,
                "match_label": b.match_label,
                "outcome_label": b.outcome_label,
                "model_prob": b.model_prob,
                "odds_taken": b.odds_taken,
                "stake": b.stake,
                "placed_at": b.placed_at.isoformat(),
                "closing_odds": b.closing_odds,
                "actual_outcome": b.actual_outcome,
                "settled_at": b.settled_at.isoformat() if b.settled_at else None,
            }
            for b in journal.bets
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_journal(path: str) -> PaperTradingJournal:
    """Returns an empty journal if `path` doesn't exist yet -- a fresh journal
    on first use is the normal case, not an error."""
    if not os.path.exists(path):
        return PaperTradingJournal()

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    bets = [
        PaperBet(
            bet_id=d["bet_id"],
            match_label=d["match_label"],
            outcome_label=d["outcome_label"],
            model_prob=d["model_prob"],
            odds_taken=d["odds_taken"],
            stake=d["stake"],
            placed_at=datetime.fromisoformat(d["placed_at"]),
            closing_odds=d.get("closing_odds"),
            actual_outcome=d.get("actual_outcome"),
            settled_at=datetime.fromisoformat(d["settled_at"]) if d.get("settled_at") else None,
        )
        for d in data.get("bets", [])
    ]
    return PaperTradingJournal(bets=bets)
