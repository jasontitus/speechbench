#!/usr/bin/env python3
"""Pre-render static HTML pages per language for SEO/crawlability.

Reads public/results.json and public/index.html (the SPA template),
generates one static page per language at public/<lang>/index.html
with:
  - Pre-rendered table rows (no JS needed for content)
  - Correct <title>, <meta description>, <link rel="canonical">
  - Open Graph tags for social sharing
  - The same JS still works for sorting/interaction

Run before `firebase deploy`:
    python web/build.py
    firebase deploy --only hosting
"""
import json
import os
import re
from pathlib import Path

WEB_DIR = Path(__file__).parent
PUBLIC = WEB_DIR / "public"
RESULTS = PUBLIC / "results.json"
TEMPLATE = PUBLIC / "index.html"
BASE_URL = "https://speechbench-viz.web.app"


def fmt_pct(x):
    if x is None:
        return ""
    return f"{x * 100:.2f}%"


def fmt_num(x):
    if x is None:
        return ""
    return f"{x:.1f}"


def fmt_int(x):
    if x is None:
        return ""
    return str(round(x))


def render_table_rows(results, datasets, dataset_specs):
    """Render the full results HTML for one language (all datasets)."""
    html_parts = []
    for ds_key in datasets:
        spec = dataset_specs.get(ds_key, {})
        title = spec.get("title", ds_key)
        url = spec.get("url", "#")
        rows = results.get(ds_key, [])
        if not rows:
            continue

        # Sort by WER
        rows_sorted = sorted(rows, key=lambda r: r.get("wer", 999))
        best_wer = rows_sorted[0].get("wer") if rows_sorted else None

        html_parts.append(f'<section class="dataset-section">')
        html_parts.append(
            f'<h2><a href="{url}" target="_blank" rel="noopener">{title}</a>'
            f'<span class="ds-key">{ds_key}</span></h2>'
        )
        html_parts.append('<div class="table-wrap"><table><thead><tr>')
        cols = [
            ("Model", "model"), ("Backend", ""), ("n", ""),
            ("WER", "wer"), ("CER", ""), ("RTFx mean", ""),
            ("RTFx p50", ""), ("Lat mean (ms)", ""),
            ("Lat p90 (ms)", ""), ("GPU peak (MB)", ""), ("Wall (s)", ""),
        ]
        for label, cls in cols:
            c = f' class="{cls}"' if cls else ""
            html_parts.append(f"<th{c}>{label}</th>")
        html_parts.append("</tr></thead><tbody>")

        for row in rows_sorted:
            wer = row.get("wer")
            is_best = best_wer is not None and wer is not None and abs(wer - best_wer) < 1e-9
            cls = ' class="best"' if is_best else ""
            if wer and wer > 1.0:
                cls = ' class="hallucinate"'

            model_key = row.get("model_key", "")
            model_url = row.get("model_url", "")
            model_cell = (
                f'<a href="{model_url}" target="_blank" rel="noopener">{model_key}</a>'
                if model_url else model_key
            )

            html_parts.append(f"<tr{cls}>")
            html_parts.append(f'<td class="model">{model_cell}</td>')
            html_parts.append(f"<td>{row.get('backend', '')}</td>")
            html_parts.append(f"<td>{fmt_int(row.get('n'))}</td>")
            html_parts.append(f'<td class="wer">{fmt_pct(row.get("wer"))}</td>')
            html_parts.append(f"<td>{fmt_pct(row.get('cer'))}</td>")
            html_parts.append(f"<td>{fmt_num(row.get('rtfx_mean'))}</td>")
            html_parts.append(f"<td>{fmt_num(row.get('rtfx_p50'))}</td>")
            html_parts.append(f"<td>{fmt_int(row.get('latency_ms_mean'))}</td>")
            html_parts.append(f"<td>{fmt_int(row.get('latency_ms_p90'))}</td>")
            html_parts.append(f"<td>{fmt_int(row.get('gpu_peak_mem_mb'))}</td>")
            html_parts.append(f"<td>{fmt_int(row.get('wall_time_s'))}</td>")
            html_parts.append("</tr>")

        html_parts.append("</tbody></table></div></section>")

    return "\n".join(html_parts)


def build():
    data = json.loads(RESULTS.read_text())
    template = TEMPLATE.read_text()
    dataset_specs = data.get("datasets", {})

    for lang_code, lang_info in data.get("languages", {}).items():
        label = lang_info.get("label", lang_code)
        results = lang_info.get("results", {})
        dataset_order = lang_info.get("dataset_order", list(results.keys()))

        # Pre-render the tables
        tables_html = render_table_rows(results, dataset_order, dataset_specs)

        # Build the page from the template
        page = template

        # Inject SEO meta tags before </head>
        meta = f"""
    <title>speechbench — {label} ASR benchmarks</title>
    <meta name="description" content="Comparative benchmarks for speech-to-text models on {label} datasets. WER, CER, RTFx, latency, and GPU memory for Whisper, Parakeet, and more.">
    <link rel="canonical" href="{BASE_URL}/{lang_code}/">
    <meta property="og:title" content="speechbench — {label}">
    <meta property="og:description" content="ASR model benchmarks for {label}: WER, CER, speed, and memory across {len(results)} datasets.">
    <meta property="og:url" content="{BASE_URL}/{lang_code}/">
    <meta property="og:type" content="website">"""
        page = page.replace("</head>", meta + "\n</head>")

        # Remove existing <title> if any
        page = re.sub(r"<title>[^<]*</title>\s*", "", page, count=1)

        # Inject pre-rendered content as a <noscript> fallback +
        # a hidden div that JS can replace
        prerendered = f"""
<noscript>
<div id="prerendered-content">
{tables_html}
</div>
</noscript>
<!-- Pre-rendered tables for crawlers (JS replaces this on load) -->
<div id="static-tables" style="display:none" data-lang="{lang_code}">
{tables_html}
</div>"""
        page = page.replace('<div id="main">', f'{prerendered}\n<div id="main">')

        # Set the default language to this page's language
        page = page.replace(
            'let currentLang = (location.hash || "").replace("#", "") || "en";',
            f'let currentLang = "{lang_code}";',
        )

        # Write the page
        out_dir = PUBLIC / lang_code
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "index.html").write_text(page)
        print(f"  {lang_code}/ — {label} ({len(results)} datasets)")

    print(f"\nBuilt {len(data['languages'])} language pages")


if __name__ == "__main__":
    build()
