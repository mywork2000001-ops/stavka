from bukmeker.coupon import ValueBetCandidate, combo_is_valid, generate_coupons, pairwise_correlation


def make_bet(bet_id, match_id, league_id, team_ids, prob, odds):
    return ValueBetCandidate(
        bet_id=bet_id, match_id=match_id, league_id=league_id, team_ids=team_ids, prob=prob, odds=odds
    )


def test_pairwise_correlation_same_match_is_maximal():
    a = make_bet(1, match_id=100, league_id=1, team_ids=(1, 2), prob=0.5, odds=2.0)
    b = make_bet(2, match_id=100, league_id=1, team_ids=(1, 2), prob=0.5, odds=2.0)
    assert pairwise_correlation(a, b) == 1.0


def test_pairwise_correlation_shared_team_moderate():
    a = make_bet(1, match_id=100, league_id=1, team_ids=(1, 2), prob=0.5, odds=2.0)
    b = make_bet(2, match_id=101, league_id=1, team_ids=(2, 3), prob=0.5, odds=2.0)
    assert pairwise_correlation(a, b) == 0.6


def test_pairwise_correlation_independent_matches_is_zero():
    a = make_bet(1, match_id=100, league_id=1, team_ids=(1, 2), prob=0.5, odds=2.0)
    b = make_bet(2, match_id=200, league_id=2, team_ids=(3, 4), prob=0.5, odds=2.0)
    assert pairwise_correlation(a, b) == 0.0


def test_combo_is_valid_rejects_same_match_pair():
    a = make_bet(1, 100, 1, (1, 2), 0.5, 2.0)
    b = make_bet(2, 100, 1, (1, 2), 0.5, 2.0)
    assert not combo_is_valid((a, b), max_corr=0.3)


def test_generate_coupons_only_returns_positive_ev_combos():
    bets = [
        make_bet(1, 100, 1, (1, 2), prob=0.55, odds=2.0),  # EV = 0.1
        make_bet(2, 101, 2, (3, 4), prob=0.55, odds=2.0),
        make_bet(3, 102, 3, (5, 6), prob=0.30, odds=2.0),  # EV = -0.4, negative
    ]
    coupons = generate_coupons(bets, bankroll=1000, max_events=2, max_corr=0.3)
    assert len(coupons) > 0
    assert all(c["ev"] > 0 for c in coupons)


def test_generate_coupons_excludes_correlated_combos():
    bets = [
        make_bet(1, 100, 1, (1, 2), prob=0.6, odds=2.0),
        make_bet(2, 100, 1, (1, 2), prob=0.6, odds=2.0),  # same match as bet 1
    ]
    coupons = generate_coupons(bets, bankroll=1000, max_events=2, max_corr=0.3)
    assert all(c["n_legs"] == 1 for c in coupons)  # the 2-leg combo must be filtered out


def test_generate_coupons_respects_top_n():
    bets = [make_bet(i, 100 + i, i, (i,), prob=0.6, odds=2.0) for i in range(10)]
    coupons = generate_coupons(bets, bankroll=1000, max_events=1, max_corr=0.3, top_n=3)
    assert len(coupons) == 3
