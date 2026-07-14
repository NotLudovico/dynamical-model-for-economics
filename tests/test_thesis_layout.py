from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_thesis_figure_generators_write_to_the_assets_folder():
    mean_degree_source = (ROOT / "scripts" / "thesis_meandeg_figures.py").read_text()
    correlation_source = (ROOT / "scripts" / "between_firm_correlations.py").read_text()

    assert 'os.path.join(ROOT, "thesis", "assets", "figures")' in mean_degree_source
    assert '"thesis/assets/figures/between_firm_corr.png"' in correlation_source
