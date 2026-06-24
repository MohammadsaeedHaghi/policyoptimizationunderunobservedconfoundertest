"""ExperimentConfig: JSON round-trip, validation guards, method coercion, Kallus flat variant."""
import pytest

from common.config import ExperimentConfig, GeometryConfig, SupportConfig, MethodSpec


def test_json_roundtrip(tmp_path):
    cfg = ExperimentConfig(
        experiment="t", n_arms=2, cap=(1.0, 0.4), gamma_true=3.0, gammas=(1.0, 3.0), seeds=(0, 1),
        methods=["IPW-O-W", {"name": "Hajek-O-W", "variants": ["Capped"]}, "Kallus"],
        geometry=GeometryConfig(zscore=True), support=SupportConfig(snap=True, mesh=5))
    p = cfg.save_json(tmp_path / "c.json")
    cfg2 = ExperimentConfig.load_json(p)
    assert cfg2.to_dict() == cfg.to_dict()


def test_kallus_forced_flat():
    assert MethodSpec("Kallus", variants=("Capped", "Uncapped")).variants == ("Flat",)


def test_method_coercion():
    cfg = ExperimentConfig(experiment="t", n_arms=2, cap=(1.0, 0.4), gamma_true=3.0, gammas=(1.0,),
                           seeds=(0,), methods=["IPW-X-X", MethodSpec("IPW-O-W"), {"name": "IPW-O-X", "variants": ["Capped"]}])
    assert [m.name for m in cfg.methods] == ["IPW-X-X", "IPW-O-W", "IPW-O-X"]
    assert cfg.methods[2].variants == ("Capped",)


@pytest.mark.parametrize("bad", [
    lambda: GeometryConfig(metric="manhattan"),
    lambda: MethodSpec("RW"),
    lambda: MethodSpec("IPW-O-W", ("Nope",)),
    lambda: ExperimentConfig(experiment="t", n_arms=3, cap=(1.0, 0.4), gamma_true=3.0,
                             gammas=(1.0,), seeds=(0,), methods=["IPW-X-X"]),   # cap length != n_arms
])
def test_validation_guards(bad):
    with pytest.raises((ValueError, TypeError)):
        bad()
