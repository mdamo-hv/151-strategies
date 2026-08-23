from __future__ import annotations

from strategies151.backtest.windows import make_folds


def test_train_and_test_lengths_are_exact():
    folds = make_folds(2000, train_days=252, test_days=21)
    assert folds
    assert all(f.train_length == 252 for f in folds)
    assert all(f.test_length == 21 for f in folds)


def test_test_windows_are_contiguous_and_disjoint():
    folds = make_folds(2000, train_days=252, test_days=21)
    for previous, current in zip(folds, folds[1:]):
        assert current.test_start == previous.test_end


def test_training_window_immediately_precedes_test_window():
    for fold in make_folds(2000, train_days=252, test_days=21):
        assert fold.train_end == fold.test_start


def test_warmup_pushes_the_first_test_window_out():
    plain = make_folds(2000, train_days=252, test_days=21, warmup=0)
    warmed = make_folds(2000, train_days=252, test_days=21, warmup=300)
    assert warmed[0].test_start == plain[0].test_start + 300
    assert warmed[0].train_start >= 300


def test_no_fold_runs_past_the_end_of_the_data():
    n = 1000
    for fold in make_folds(n, train_days=252, test_days=21):
        assert fold.test_end <= n
        assert fold.train_start >= 0


def test_short_history_yields_no_folds():
    assert make_folds(100, train_days=252, test_days=21) == []


def test_step_larger_than_test_leaves_gaps():
    folds = make_folds(2000, train_days=252, test_days=21, step_days=63)
    assert folds[1].test_start - folds[0].test_end == 42
