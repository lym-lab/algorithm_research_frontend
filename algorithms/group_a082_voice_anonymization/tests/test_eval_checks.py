"""Internal checks and the qualitative verdicts of rows 6-8, on made-up averages."""
from a082.eval import targets


def m(eer, uar, wer, dev_f=None):
    return {"EER_avg": eer, "UAR_avg": uar, "WER_avg": wer, "EER": {"dev_f": dev_f} if dev_f is not None else {}}


def test_row8_is_judged_against_row7():
    ours = {"5": m(8.5, 54.8, 3.0), "7": m(13.7, 43.6, 10.2)}
    assert targets.qualitative("8", m(13.9, 43.5, 10.8), ours) == "qualitative ok"
    assert targets.qualitative("8", m(12.0, 43.5, 10.8), ours) == "qualitative NOT met"   # below row 7
    assert targets.qualitative("8", m(13.9, 43.5, 12.0), ours) == "qualitative NOT met"   # WER not close
    assert targets.qualitative("8", m(13.9, 43.5, 10.8), {"5": ours["5"]}) is None        # row 7 missing


def test_internal_checks_paper_numbers_pass():
    # The paper's own Table 1 averages satisfy every check.
    ours = {r: m(e, u, w) for r, (_, e, u, w) in targets.TABLE1.items()}
    sweep = {0.1: m(None, 54.0, 3.2, 15.0), 0.2: m(None, 51.7, 3.3, 25.0), 0.5: m(None, 44.9, 3.4, 47.0)}
    res = dict(targets.internal_checks(ours, sweep))
    assert len(res) == 6 and all(v.startswith("ok") for v in res.values()), res


def test_internal_checks_catch_linear_mix_and_non_monotone_sweep():
    ours = {r: m(e, u, w) for r, (_, e, u, w) in targets.TABLE1.items()}
    ours["mix0.2"] = m(16.0, 51.4, 3.28)          # EER only at the linear interpolation: no poisoning effect
    sweep = {0.1: m(None, 54.0, 3.2, 25.0), 0.2: m(None, 51.7, 3.3, 15.0), 0.3: m(None, 49.0, 3.3, 30.0)}
    res = dict(targets.internal_checks(ours, sweep))
    assert res["mix0.2 vs linear mix of rows 1, 4"].startswith("NOT met")
    assert res["Fig. 4: dev-f EER rises and UAR falls with p"] == "NOT met"


def test_tradeoff_figure():
    from a082 import viz
    fig = viz.tradeoff_figure({0.1: (15.0, 54.0), 0.5: (47.0, 44.8)}, {"kNN-VC (1)": (11.8, 56.7)})
    assert len(fig.axes) == 1
