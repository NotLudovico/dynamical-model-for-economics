# Thesis folder

The thesis directory is organised by purpose:

- `main.tex`, `bib.bib`, `build.sh`: main thesis entry points.
- `chapters/`: chapter and appendix source files.
- `assets/figures/`: every image used by the thesis or retained for historical reference.
- `build/`: LaTeX auxiliary files and the build copy of `main.pdf`.
- `slides/`: presentation source and final PDF; its auxiliary files live in `slides/build/`.
- `standalone/`: documents that compile independently of the main thesis.
- `FIGURES.md`: provenance and regeneration instructions for the figures.

Build the main thesis from the repository root with:

```sh
bash thesis/build.sh
```

The build intermediates go to `thesis/build/`. The final PDF is copied to
`thesis/main.pdf` for convenient access.

Standalone documents are grouped as follows:

- `standalone/abstract/`
- `standalone/dmft_theory/`
- `standalone/zero_growth_theory/`

Each standalone source should be compiled from its own directory so its relative
bibliography and figure paths resolve correctly. Keep auxiliary files in the central
build tree, for example:

```sh
cd thesis/standalone/zero_growth_theory
latexmk -pdf -outdir=../../build/standalone/zero_growth_theory zero_growth_theory.tex
```

For the presentation:

```sh
cd thesis/slides
latexmk -pdf -outdir=build slides.tex
cp build/slides.pdf slides.pdf
```
