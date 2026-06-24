"""The clean guard: prune_experiment_dir removes stale method outputs, keeps run-level + planned."""
from common.persistence import prune_experiment_dir


def _build(exp):
    leaves = ["IPW-O-W/Capped/gtrue-3", "IPW-O-W/Uncapped/gtrue-3", "IPW-O-X/Capped/gtrue-3",
              "Kallus/Flat/gtrue-3", "IPW-O-W/Capped/gtrue-5"]
    for leaf in leaves:
        (exp / leaf).mkdir(parents=True)
        (exp / leaf / "outcomes.csv").write_text("x")
    for run in ("data", "diagnostics", "reference"):
        (exp / run).mkdir(parents=True)
        (exp / run / "f.json").write_text("x")
    (exp / "config.json").write_text("{}")


def test_dry_run_reports_without_deleting(tmp_path):
    exp = tmp_path / "exp"; _build(exp)
    planned = [exp / "IPW-O-W/Capped/gtrue-3", exp / "Kallus/Flat/gtrue-3"]
    would = prune_experiment_dir(exp, planned, dry_run=True)
    assert len(would) == 3                                # IPW-O-W/Uncapped, IPW-O-X/Capped, IPW-O-W/Capped/gtrue-5
    assert (exp / "IPW-O-X").exists()                         # nothing actually removed


def test_prune_removes_only_orphans(tmp_path):
    exp = tmp_path / "exp"; _build(exp)
    planned = [exp / "IPW-O-W/Capped/gtrue-3", exp / "Kallus/Flat/gtrue-3"]
    removed = prune_experiment_dir(exp, planned, dry_run=False)
    assert len(removed) == 3
    # planned kept
    assert (exp / "IPW-O-W/Capped/gtrue-3").exists()
    assert (exp / "Kallus/Flat/gtrue-3").exists()
    # orphans gone (dropped method, dropped variant, superseded gamma_true)
    assert not (exp / "IPW-O-X").exists()
    assert not (exp / "IPW-O-W/Uncapped").exists()
    assert not (exp / "IPW-O-W/Capped/gtrue-5").exists()
    # run-level + config never touched
    assert all((exp / d).exists() for d in ("data", "diagnostics", "reference"))
    assert (exp / "config.json").exists()
