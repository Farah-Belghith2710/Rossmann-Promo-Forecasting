"""Write the results section of the README from the metrics, between two marker lines."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

START = "<!-- results:start -->"
END = "<!-- results:end -->"

_CHARTS = [
    ("forecast_total", "Daily sales of all stores over the test weeks: actual sales, LightGBM forecast and median baseline"),
    ("promo_uplift", "Extra sales on promotion days, by store type and by weekday"),
]


def _day(iso: str) -> str:
    d = dt.date.fromisoformat(iso)
    return f"{d.day} {d:%B %Y}"


def _picture(folder: str, name: str, alt: str) -> str:
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="{folder}/{name}-dark.png">\n'
            f'  <img alt="{alt}" src="{folder}/{name}.png">\n</picture>')


def results_markdown(metrics: dict, reports_folder: str = "reports") -> str:
    base = metrics["accuracy_on_test_weeks"]["median_baseline"]
    lgb = metrics["accuracy_on_test_weeks"]["lightgbm"]
    uplift = metrics["promo_uplift"]["overall"]["model_uplift_pct"]
    lower = 100 * (1 - lgb["rmspe"] / base["rmspe"])
    return "\n".join([
        f"- The forecast's error is about **{lower:.0f}% lower** than a simple baseline's "
        f"({lgb['rmspe']:.3f} vs {base['rmspe']:.3f} RMSPE).",
        f"- Promotions raise sales by about **{uplift:.0f}%** on promotion days.",
        "",
        f"![Daily sales of all stores: actual sales vs forecast]({reports_folder}/figures/forecast_total.png)",
    ])


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
