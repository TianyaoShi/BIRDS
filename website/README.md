# BIRDS Project Page

A static, responsive research page for GitHub Pages. All figure interactions run
in the browser. No server, API keys, analytics, or raw research datasets are needed.

## Development

Requires Node.js 22.12+ (or 24) and npm.

```sh
cd website
npm ci
npm run dev
```

```sh
npm run build
npx playwright install chromium
npm test
```

Tests serve the production build under `/BIRDS/`, exercise all workload choices,
check charts for nonblank pixels, test desktop/mobile layouts, and save screenshots
to the ignored `test-output/` folder. `npm run preview` previews the build.

## Content and Data

- `index.html`: narrative, authors, methods, and citation.
- `src/main.js`: four interactive comparisons using Chart.js.
- `src/sources.js`: the 12 sources cited in Appendix B.3 of arXiv v3.
- `src/results.json`: approved processed figure values and source-file checksums.
- `src/style.css`: responsive presentation and accessibility states.

The results are pinned to **arXiv:2605.27480v3 (August 29, 2026)** and exported
from the preserved `visualization_emnlp/derived` archive. Regeneration is an offline
maintainer operation, never part of the public build:

```sh
python website/scripts/export_results.py /path/to/visualization_emnlp/derived
```

The exporter uses a field allowlist and validates BI / normalized quality = QNBI,
finite nonnegative values, unique models, and contribution-share totals. It excludes
raw energy, profiling traces, environmental intensities, filesystem paths, and job IDs.
Published values are downloadable by anyone visiting the page.

The public interface emphasizes visual comparisons: it omits result tables,
numerical value-axis labels, and numerical result tooltips. Model identifiers,
assessment horizons, units, and scale descriptions remain visible. This is a
presentation choice, not a restriction on access to the bundled processed data.

| Page view | Paper | Published data |
| --- | --- | --- |
| Composition | Figure 4 | Mean lifecycle shares and horizon-specific pathway shares |
| Model comparison | Figures 5-6, 11-26 | 137 selected model/workload rows; BI, QNBI, quality, model and GPU labels |
| Reasoning | Figure 8 | Six Instruct/Thinking rows; BI, QNBI and quality |
| Hardware | Figures 10, 27 | 24 model/GPU rows; BI, QNBI and quality |

The composition view is an explicitly labeled mean-share adaptation of the paper's
lifecycle boxplots. The page does not reproduce response-length distributions or
throughput overlays. QNBI workload files include only models with quality scores;
the page does not silently substitute values for missing model/workload pairs.
The archived selection includes some A100 supplements to H100 results. Archived
figure inputs record incremental GPU energy; that basis is disclosed on the page.

Verification: the archived Figure 4 global-warming shares round to the PDF's
71.1%, 89.1%, and 98.7% for the 20/100/1000-year horizons. Reasoning quality
scores agree with Table 10. Figure input hashes identify the exact exported files.

## Publishing

The `.github/workflows/pages.yml` workflow builds and browser-tests the site and
uploads **only `website/dist`**. Pull requests run validation without deployment.
On GitHub, choose **Settings > Pages > Build and deployment > GitHub Actions**.
Push changes to `main` or run the workflow manually. The intended URL is:

https://tianyaoshi.github.io/BIRDS/

The build has an explicit `/BIRDS/` base path. Update the build script, metadata,
and test base path when moving to a different repository or domain.

## Asset Credits

- Forest photograph: Sebastian Unrau, [Unsplash](https://unsplash.com/photos/trees-on-forest-with-sun-rays-sp-p7uuT0tw), distributed under the [Unsplash License](https://unsplash.com/license). This is illustrative habitat imagery, not a studied location.
- Dove and interface icons: Font Awesome Free 5.15.4; SVG icons CC BY 4.0, fonts SIL OFL 1.1, code MIT. See `public/ASSET-LICENSES.txt`.
- Chart.js: MIT; dependency license retained by the package distribution.
- Website code follows the repository's Apache-2.0 license. Third-party asset licenses remain separate.
