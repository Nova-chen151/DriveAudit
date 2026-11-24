# README figure and result provenance

The root [README](<../../README.md>) uses one framework figure rendered from the user-supplied `ICLR2027_Benchmark.zip` manuscript sources and an original animated project wordmark. The framework is a manuscript illustration, not an output produced by running this checkout. No SimScale images, results, author information, or release claims are reused.

## Manuscript

- Title: **Beyond Visual Realism: Benchmarking Generated Driving Scenarios as Policy Tests**.
- Benchmark name: **DriveAudit**.
- Author field: **Anonymous authors**.
- Source: `iclr2027_conference.tex` in the supplied archive.
- The archive contains LaTeX sources and individual figure PDFs, not a compiled full-paper PDF or a confirmed public paper URL.

## Framework figure

[overview.png](<overview.png>) is the first page of `figs/overview.pdf` in the supplied archive, rasterized on a white background at 2,200 pixels wide. Its content is unchanged. This is the only paper figure retained in the README; the PDF source is not duplicated in this repository.

## Animated title

[title.gif](<title.gif>) is an original DriveAudit wordmark, not a modified SimScale asset. Blue driving-trajectory motion forms a returning feedback path beneath a bold italic wordmark, with a single moving highlight across the lettering. The first frame is fully readable even before animation begins.

- Canvas: 1,600 × 360 pixels, displayed responsively in the README.
- Timing: 72 source frames at 70 ms each, with a short resting phase; approximately five seconds per loop. The GIF encoder merges identical hold frames into longer-duration frames.
- Rendering: a shared indexed palette prevents frame-to-frame color flicker; no JavaScript, external image hosting, or CSS animation is needed on GitHub.
- Font: an installed Avenir Next Heavy Italic face. The font file is not redistributed. A compatible installed font can be supplied on other platforms.
- Accessibility: the image includes a descriptive alt attribute. GIF playback preferences are controlled by the Markdown host/browser; a GIF itself cannot implement a CSS reduced-motion query.

Regenerate from the repository root using the [Pillow renderer](<render_title.py>):

```bash
python assets/readme/render_title.py
# Other platforms: pass a compatible installed bold italic font.
python assets/readme/render_title.py --font /path/to/your/font.ttf
```

## Numerical results and scope

Both main comparison tables in the root README are transcribed from the table labeled `tab:main` in `iclr2027_conference.tex`:

- **Fixed recorded trajectories + rendering**: seven generator rows with shared `TRS = 1.000`.
- **Generated trajectories + rendering**: seven generator rows with shared `TRS = 0.782`.
- **Real video**: the reference row, with `QTES = 50.000`.

These are manuscript-reported results, not independently reproduced measurements. The main study uses 550 scenarios and six policies under paired non-reactive replay. The manuscript's additional 20-scenario closed-loop study is a different evaluation subset and protocol; its scores must not be mixed into the main tables. The supplied closed-loop table also contains a comment noting a pending route-projection validity audit, so its numerical results are intentionally not promoted into this README.

The repository's two-scene nuScenes example and DrivoR closed-loop runner are documented separately from those manuscript-level claims. Public paper links, author identities, release dates, dataset download endpoints, and final BibTeX are omitted until confirmed. The source archive itself is not committed or linked as a public download.

## Rights

The supplied paper figures remain subject to their original rights and source-data terms. Rendering them for the project README does not relicense source datasets, upstream code, or model weights.
