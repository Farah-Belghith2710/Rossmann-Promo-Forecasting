"""Write the results section of the README from the metrics, between two marker lines."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

START = "<!-- results:start -->"
END = "<!-- results:end -->"

_CHARTS = [
    ("forecast_total", "Daily sales of all stores over the test weeks: actual sales, LightGBM forecast and median baseline"),
    ("forecast_stores", "Actual sales and LightGBM forecast for a small, a typical and a large store"),
    ("promo_uplift", "Extra sales on promotion days, by store type and by weekday"),
    ("feature_importance", "The 12 features the model relies on most"),
]


def _day(iso: str) -> str:
    d = dt.date.fromisoformat(iso)
    return f"{d.day} {d:%B %Y}"


def _picture(folder: str, name: str, alt: str) -> str:
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="{folder}/{name}-dark.png">\n'
            f'  <img alt="{alt}" src="{folder}/{name}.png">\n</picture>')


def results_markdown(metrics: dict, reports_folder: str = "reports") -> str:
    acc, up, data, split = (metrics["accuracy_on_test_weeks"], metrics["promo_uplift"], metrics["data"],
                            metrics["split"])
    first, last = split["test_period"].split(" to ")
    base, lgb = acc["median_baseline"], acc["lightgbm"]
    by_type = sorted(up["by_store_type"], key=lambda r: r["model_uplift_pct"])
    overall = up["overall"]
    weekdays = up["weekdays_used"]
    days = "Monday–Friday" if weekdays == [1, 2, 3, 4, 5] else f"weekdays {weekdays}"

    figures_folder = f"{reports_folder}/figures"
    lines = [
        f"Measured on the test weeks, {_day(first)} to {_day(last)}, which the model never saw during training "
        f"({split['test_store_days']:,} store-days, {split['test_stores']:,} stores):",
        "",
        "| Forecast | RMSPE | MAPE |",
        "|---|---|---|",
        f"| Median baseline: past median of the same store, weekday and promo status | {base['rmspe']:.3f} | "
        f"{100 * base['mape']:.1f}% |",
        f"| **LightGBM** (this project) | **{lgb['rmspe']:.3f}** | **{100 * lgb['mape']:.1f}%** |",
        "",
        "RMSPE is the error measure of the Kaggle competition: the typical error in percent of the real sales. "
        f"Lower is better. LightGBM cuts the baseline's error by {100 * (1 - lgb['rmspe'] / base['rmspe']):.0f}%.",
        "",
        f"On a promotion day, a store sells about **{overall['model_uplift_pct']:.0f}% more** than it would without "
        f"the promotion, about €{overall['extra_sales_per_store_day']:,.0f} extra per store and day. The effect "
        f"depends on the store: from +{by_type[0]['model_uplift_pct']:.0f}% for type {by_type[0]['StoreType']} "
        f"stores to +{by_type[-1]['model_uplift_pct']:.0f}% for type {by_type[-1]['StoreType']} stores "
        f"({days} store-days of the test weeks).",
        "",
        _picture(figures_folder, *_CHARTS[0]),
        "",
        _picture(figures_folder, *_CHARTS[1]),
        "",
        _picture(figures_folder, *_CHARTS[2]),
        "",
        "| Store type | Store-days | Uplift (model) | Extra sales per store-day | Plain difference in the data |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(up["by_store_type"], key=lambda r: r["StoreType"]):
        lines.append(f"| {r['StoreType']} | {int(r['store_days']):,} | +{r['model_uplift_pct']:.1f}% | "
                     f"€{r['extra_sales_per_store_day']:,.0f} | +{r['observed_uplift_pct']:.1f}% |")
    lines += [
        f"| **all** | {int(overall['store_days']):,} | **+{overall['model_uplift_pct']:.1f}%** | "
        f"€{overall['extra_sales_per_store_day']:,.0f} | +{overall['observed_uplift_pct']:.1f}% |",
        "",
        _picture(figures_folder, *_CHARTS[3]),
        "",
        f"Data: {data['rows']:,} rows, {data['stores']:,} stores, {_day(data['first_day'])} to {_day(data['last_day'])}; "
        f"{data['stores_with_missing_days']} stores have missing days. Full run: {metrics['runtime_seconds'] / 60:.0f} "
        f"minutes. All the numbers are in [`metrics.json`]({reports_folder}/metrics.json).",
    ]
    return "\n".join(lines)


def update_readme(metrics: dict, readme: str | Path, reports_dir: str | Path) -> bool:
    """Replace the text between the markers. Returns False if the README has no markers."""
    readme = Path(readme)
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return False
    reports_folder = Path(os.path.relpath(Path(reports_dir), readme.parent)).as_posix()
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    readme.write_text(f"{before}{START}\n{results_markdown(metrics, reports_folder)}\n{END}{after}", encoding="utf-8")
    return True
