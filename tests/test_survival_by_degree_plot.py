import matplotlib.pyplot as plt
import importlib


def test_configure_survival_axes_focuses_on_resolved_degree_and_probability_ranges():
    from scripts.survival_by_degree import configure_survival_axes

    figure, axis = plt.subplots()
    configure_survival_axes(axis, degree=[75.0, 126.0, 210.0], ci_high=[0.208, 0.162, 0.107])

    assert axis.get_xscale() == "log"
    assert axis.get_xlim()[0] < 75.0
    assert axis.get_ylim()[1] < 0.3
    plt.close(figure)


def test_survival_study_defaults_to_the_legacy_degree_distribution(monkeypatch):
    monkeypatch.delenv("SBD_MIN_DEGREE", raising=False)
    import scripts.survival_by_degree as study

    study = importlib.reload(study)

    assert study.MIN_DEGREE is None
