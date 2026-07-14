from matplotlib.mathtext import MathTextParser

from scripts.plot_hdmft_zero_growth import OUTPUT_DIR, ROOT, UNSTABLE_LABEL


def test_unstable_continuation_label_is_valid_mathtext():
    MathTextParser("agg").parse(UNSTABLE_LABEL)


def test_zero_growth_report_is_generated_in_the_thesis_standalone_folder():
    assert OUTPUT_DIR == ROOT / "thesis" / "standalone" / "zero_growth_theory"
