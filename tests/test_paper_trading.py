import math
from datetime import datetime

import pytest

from bukmeker.paper_trading import PaperTradingJournal, load_journal, save_journal


def test_log_bet_assigns_sequential_ids():
    journal = PaperTradingJournal()
    b1 = journal.log_bet("Arsenal vs Chelsea", "home_win", 0.6, 2.0, 100.0)
    b2 = journal.log_bet("Liverpool vs Everton", "home_win", 0.7, 1.5, 50.0)
    assert b1.bet_id == 1
    assert b2.bet_id == 2


def test_log_bet_rejects_invalid_odds_and_prob():
    journal = PaperTradingJournal()
    with pytest.raises(ValueError):
        journal.log_bet("A vs B", "home_win", 0.5, 1.0, 100.0)
    with pytest.raises(ValueError):
        journal.log_bet("A vs B", "home_win", 1.5, 2.0, 100.0)


def test_profit_raises_until_settled():
    journal = PaperTradingJournal()
    bet = journal.log_bet("A vs B", "home_win", 0.6, 2.0, 100.0)
    with pytest.raises(ValueError, match="not settled"):
        _ = bet.profit


def test_settle_bet_win_computes_correct_profit():
    journal = PaperTradingJournal()
    bet = journal.log_bet("A vs B", "home_win", 0.6, 2.0, 100.0)
    journal.settle_bet(bet.bet_id, won=True)
    assert bet.profit == pytest.approx(100.0)  # stake * (odds - 1)


def test_settle_bet_loss_computes_correct_profit():
    journal = PaperTradingJournal()
    bet = journal.log_bet("A vs B", "home_win", 0.6, 2.0, 100.0)
    journal.settle_bet(bet.bet_id, won=False)
    assert bet.profit == pytest.approx(-100.0)


def test_settle_unknown_bet_id_raises():
    journal = PaperTradingJournal()
    with pytest.raises(ValueError, match="no bet"):
        journal.settle_bet(999, won=True)


def test_clv_is_none_until_closing_odds_recorded():
    journal = PaperTradingJournal()
    bet = journal.log_bet("A vs B", "home_win", 0.6, 2.0, 100.0)
    assert bet.clv is None
    journal.settle_bet(bet.bet_id, won=True, closing_odds=2.2)
    assert bet.clv == pytest.approx(math.log(2.2 / 2.0))


def test_clv_negative_when_price_got_worse_than_closing():
    journal = PaperTradingJournal()
    bet = journal.log_bet("A vs B", "home_win", 0.6, 2.0, 100.0)
    journal.settle_bet(bet.bet_id, won=True, closing_odds=1.8)
    assert bet.clv < 0  # you took a worse price than the market settled on


def test_pending_and_settled_bets_partition_correctly():
    journal = PaperTradingJournal()
    b1 = journal.log_bet("A vs B", "home_win", 0.6, 2.0, 100.0)
    journal.log_bet("C vs D", "away_win", 0.4, 3.0, 50.0)
    journal.settle_bet(b1.bet_id, won=True)

    assert [b.bet_id for b in journal.settled_bets()] == [b1.bet_id]
    assert len(journal.pending_bets()) == 1


def test_summary_with_no_settled_bets_returns_none_metrics_not_zero():
    journal = PaperTradingJournal()
    journal.log_bet("A vs B", "home_win", 0.6, 2.0, 100.0)
    summary = journal.summary()
    assert summary["n_settled"] == 0
    assert summary["n_pending"] == 1
    assert summary["hit_rate"] is None
    assert summary["roi"] is None


def test_summary_computes_hit_rate_roi_and_avg_clv():
    journal = PaperTradingJournal()
    b1 = journal.log_bet("A vs B", "home_win", 0.6, 2.0, 100.0)
    b2 = journal.log_bet("C vs D", "away_win", 0.5, 2.0, 100.0)
    journal.settle_bet(b1.bet_id, won=True, closing_odds=2.2)
    journal.settle_bet(b2.bet_id, won=False, closing_odds=1.9)

    summary = journal.summary()
    assert summary["n_settled"] == 2
    assert summary["hit_rate"] == pytest.approx(0.5)
    assert summary["total_staked"] == pytest.approx(200.0)
    assert summary["total_profit"] == pytest.approx(100.0 - 100.0)  # +100 win, -100 loss
    assert summary["roi"] == pytest.approx(0.0)
    expected_avg_clv = (math.log(2.2 / 2.0) + math.log(1.9 / 2.0)) / 2
    assert summary["avg_clv"] == pytest.approx(expected_avg_clv)


def test_save_and_load_journal_round_trip(tmp_path):
    journal = PaperTradingJournal()
    b1 = journal.log_bet("A vs B", "home_win", 0.6, 2.0, 100.0, placed_at=datetime(2026, 1, 1, 12, 0))
    journal.settle_bet(b1.bet_id, won=True, closing_odds=2.1, settled_at=datetime(2026, 1, 2, 18, 0))
    journal.log_bet("C vs D", "away_win", 0.4, 3.0, 50.0, placed_at=datetime(2026, 1, 3, 9, 0))

    path = str(tmp_path / "journal.json")
    save_journal(journal, path)
    reloaded = load_journal(path)

    assert len(reloaded.bets) == 2
    assert reloaded.bets[0].match_label == "A vs B"
    assert reloaded.bets[0].is_settled
    assert reloaded.bets[0].profit == pytest.approx(100.0)
    assert reloaded.bets[0].placed_at == datetime(2026, 1, 1, 12, 0)
    assert not reloaded.bets[1].is_settled


def test_load_journal_returns_empty_journal_when_file_missing(tmp_path):
    journal = load_journal(str(tmp_path / "does_not_exist.json"))
    assert journal.bets == []
