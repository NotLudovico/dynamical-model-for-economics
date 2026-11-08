"""Read Compustat's rescaled-growth densities off the vector graphics of Moran, Secchi & Bouchaud (arXiv:2404.15226v3).

    python scripts/msb_curves.py path/to/2404.15226v3.pdf      ->  data/msb_rescaled_growth.npz

Fig. 5 (page 20, right panel, light blue): growth rates rescaled by leave-one-out mean and volatility, whole sample.
Fig. 7 (page 24, left panel): the same by 25 size bins, drawn in bin order with a colour ramp; a curve that leaves the
axes at the bottom is split into several paths of one colour.  Both are kernel density estimates.  The curves are read
from the paths rather than rebuilt from the GSE parameters of MSB's Table 1, which do not reproduce the plotted fit.
Axis calibration (path coordinates of the gridlines, found in the SVG): Fig. 5 x = -8..8 at 27.4..352.37, y = 1e-5..1 at
201.6..21.99; Fig. 7 x = -8..8 at 32.2..352.4, y = 1e-4..1 at 201.6..4.85.  Needs pdftocairo (poppler).
"""
import re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data" / "msb_rescaled_growth.npz"
GRID = np.linspace(-8, 8, 801)


def colour_paths(pdf, page, tx):
    """Coloured stroked paths of one panel (identified by its transform's x offset), in drawing order."""
    with tempfile.TemporaryDirectory() as tmp:
        svg = Path(tmp) / "p.svg"
        subprocess.run(["pdftocairo", "-svg", "-f", str(page), "-l", str(page), pdf, str(svg)], check=True)
        s = svg.read_text()
    out = []
    for p in re.findall(r"<path[^>]*>", s):
        tr, col = re.search(r'transform="matrix\(([^)]*)\)"', p), re.search(r'stroke="([^"]*)"', p)
        if tr and tx in tr.group(1) and col and not col.group(1).startswith("rgb(0%, 0%, 0%"):
            a = np.array(re.findall(r"[ML] ([-\d.]+) ([-\d.]+)", re.search(r' d="([^"]*)"', p).group(1)), float)
            if len(a) > 40:
                out.append((col.group(1), a))
    return out


def density(a, x0, x1, ybot, ytop, dec_bot):
    """Path coordinates -> density on GRID; zero where the curve is below the axes (clipped) or absent."""
    a = a[a[:, 1] < ybot - 0.3]
    x = -8 + 16 * (a[:, 0] - x0) / (x1 - x0)
    p = 10 ** (dec_bot - dec_bot * (ybot - a[:, 1]) / (ybot - ytop))
    o = np.argsort(x)
    return np.interp(GRID, x[o], p[o], left=0.0, right=0.0)


def main(pdf):
    whole = [a for c, a in colour_paths(pdf, 20, "0.893829") if c.startswith("rgb(33.72")]
    assert len(whole) == 1, "Fig. 5: expected one light-blue curve in the right panel"
    by_colour = {}
    for c, a in colour_paths(pdf, 24, "0.803081"):
        by_colour.setdefault(c, []).append(a)                    # dicts keep drawing order = bin order
    assert len(by_colour) == 25, f"Fig. 7: expected 25 size bins, found {len(by_colour)}"
    bins = np.array([density(np.vstack(v), 32.2, 352.4, 201.6, 4.85, -4) for v in by_colour.values()])
    np.savez_compressed(OUT, x=GRID, whole=density(whole[0], 27.4, 352.37, 201.6, 21.99, -5), bins=bins,
                        floor_whole=1e-5, floor_bins=1e-4)
    print(f"wrote {OUT}: whole-sample curve and {len(bins)} size bins on {GRID.size} points")


if __name__ == "__main__":
    main(sys.argv[1])
