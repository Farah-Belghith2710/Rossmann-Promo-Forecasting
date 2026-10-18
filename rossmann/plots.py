"""Charts for the README. Each chart is saved twice: name.png (light) and name-dark.png (dark)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "blue": "#2a78d6", "orange": "#eb6834", "actual": "#52514e"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "blue": "#3987e5", "orange": "#d95926", "actual": "#c3c2b7"},
}
DPI = 150
LINE = 1.7  # points

FEATURE_LABELS = {
    "DayOfWeek": "Day of week", "Day": "Day of month", "Month": "Month", "Year": "Year",
    "WeekOfYear": "Week of year", "DayOfYear": "Day of year", "Promo": "Promo running",
    "SchoolHoliday": "School holiday", "StateHoliday": "Public holiday type",
    "DaysToHoliday": "Days to next holiday", "DaysAfterHoliday": "Days since last holiday",
    "DaysToClosure": "Days to next closure", "DaysAfterClosure": "Days since last closure",
    "StoreType": "Store type", "Assortment": "Assortment", "CompetitionDistance": "Distance to competitor",
    "CompetitionOpenMonths": "Months since competitor opened", "CompetitionActive": "Competitor open",
    "Promo2": "In Promo2 campaign", "Promo2Weeks": "Weeks in Promo2", "IsPromo2Month": "Promo2 month",
    "StoreAvgSales": "Store's average sales", "StoreAvgCustomers": "Store's average customers",
    "StoreSalesPerCustomer": "Store's sales per customer", "StorePromoRatio": "Store's past promo lift",
    "StoreDowRatio": "Store's weekday pattern",
}
WEEKDAYS = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}


def _compact(value, _pos=None) -> str:
    """0 / 800 / 1.2K / 3M."""
    for size, suffix in ((1e6, "M"), (1e3, "K")):
        if abs(value) >= size:
            return f"{value / size:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{value:.0f}"


def _new_figure(t: dict, width: float, height: float, ncols: int = 1, **kw):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), dpi=DPI, facecolor=t["surface"], **kw)
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(t["surface"])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["axis"])
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(length=0, labelsize=9, labelcolor=t["ink2"], pad=6)
        ax.grid(axis="y", color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
    return fig, axes


def _titles(fig, t: dict, title: str, subtitle: str, top: float = 0.95) -> None:
    fig.text(0.012, top, title, fontsize=13, fontweight="bold", color=t["ink"], va="top")
    fig.text(0.012, top - 0.075, subtitle, fontsize=9.5, color=t["ink2"], va="top")


def _legend(fig, t: dict, handles, labels, y: float) -> None:
    leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.006, y), ncol=len(labels),
                     frameon=False, fontsize=9, handlelength=1.6, columnspacing=1.6)
    for text in leg.get_texts():
        text.set_color(t["ink2"])


def _rounded_bars(ax, positions, values, width: float, color: str, horizontal: bool = False,
                  radius_px: float = 6) -> None:
    """Bars with a rounded data end and a square end at the baseline."""
    ax.figure.canvas.draw()
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform((0, 0)), inv.transform((radius_px, radius_px))
    rx, ry = abs(x1 - x0), abs(y1 - y0)
    for pos, val in zip(positions, values):
        if not np.isfinite(val) or val == 0:
            continue
        s = np.sign(val)
        a, b = pos - width / 2, pos + width / 2
        if horizontal:  # length along x, thickness along y
            r_len, r_thk = min(rx, abs(val)), min(ry, width / 2)
            verts = [(0, a), (val - s * r_len, a), (val, a), (val, a + r_thk), (val, b - r_thk), (val, b),
                     (val - s * r_len, b), (0, b), (0, a)]
        else:
            r_len, r_thk = min(ry, abs(val)), min(rx, width / 2)
            verts = [(a, 0), (a, val - s * r_len), (a, val), (a + r_thk, val), (b - r_thk, val), (b, val),
                     (b, val - s * r_len), (b, 0), (a, 0)]
        codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
                 MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", zorder=2))


def _save(fig, out_dir: Path, name: str, theme: str) -> None:
    suffix = "" if theme == "light" else "-dark"
    fig.savefig(out_dir / f"{name}{suffix}.png", facecolor=fig.get_facecolor(), dpi=DPI)
    plt.close(fig)


# ---- charts --------------------------------------------------------------------------------

def forecast_total(hold: pd.DataFrame, out_dir: Path, theme: str) -> None:
    """Whole chain, day by day: actual sales vs the two forecasts."""
    t = THEMES[theme]
    daily = hold.groupby("Date")[["Sales", "LightGBM", "Baseline"]].sum()
    fig, ax = _new_figure(t, 10, 4.3)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.75, bottom=0.10)
    ax.plot(daily.index, daily["Sales"], color=t["actual"], lw=LINE, solid_capstyle="round", label="Actual sales")
    ax.plot(daily.index, daily["LightGBM"], color=t["blue"], lw=LINE, solid_capstyle="round", label="LightGBM forecast")
    ax.plot(daily.index, daily["Baseline"], color=t["orange"], lw=LINE, solid_capstyle="round", label="Median baseline")
    ax.yaxis.set_major_formatter(FuncFormatter(_compact))
    ax.set_ylim(bottom=0)
    ax.margins(x=0.01)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("Mon %d %b"))
    _titles(fig, t, "Daily sales of all stores, last 6 weeks (never seen by the model)",
            "Sum over stores, in euros. Most stores close on Sundays, hence the weekly dips.")
    _legend(fig, t, *ax.get_legend_handles_labels(), y=0.84)
    _save(fig, out_dir, "forecast_total", theme)


def forecast_stores(hold: pd.DataFrame, out_dir: Path, theme: str) -> None:
    """Three stores (small, typical, large): actual vs LightGBM, day by day."""
    t = THEMES[theme]
    avg = hold.groupby("Store")["Sales"].mean().sort_values()
    picks = [avg.index[int(q * (len(avg) - 1))] for q in (0.1, 0.5, 0.9)]
    fig, axes = _new_figure(t, 10, 3.6, ncols=3)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.64, bottom=0.14, wspace=0.18)
    for ax, (label, store) in zip(axes, zip(("Small store", "Typical store", "Large store"), picks)):
        s = hold[hold["Store"] == store].sort_values("Date")
        ax.plot(s["Date"], s["Sales"], color=t["actual"], lw=LINE - 0.2, solid_capstyle="round")
        ax.plot(s["Date"], s["LightGBM"], color=t["blue"], lw=LINE - 0.2, solid_capstyle="round")
        ax.set_title(f"{label} (no. {store}, type {s['StoreType'].iloc[0]})", fontsize=9.5, color=t["ink2"],
                     loc="left", pad=8)
        ax.yaxis.set_major_formatter(FuncFormatter(_compact))
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    _titles(fig, t, "Daily sales of three stores, last 6 weeks", "Open days only, in euros.", top=0.96)
    handles = [Line2D([], [], color=t["actual"], lw=LINE), Line2D([], [], color=t["blue"], lw=LINE)]
    _legend(fig, t, handles, ["Actual sales", "LightGBM forecast"], y=0.845)
    _save(fig, out_dir, "forecast_stores", theme)


def promo_uplift(by_type: pd.DataFrame, by_weekday: pd.DataFrame, out_dir: Path, theme: str) -> None:
    """How much a promotion adds, by store type and by weekday (model vs plain difference)."""
    t = THEMES[theme]
    fig, axes = _new_figure(t, 10, 4.2, ncols=2, gridspec_kw={"width_ratios": [4, 5]})
    fig.subplots_adjust(left=0.05, right=0.99, top=0.66, bottom=0.11, wspace=0.12)
    panels = [(axes[0], by_type, "StoreType", lambda v: f"Type {v}", "By store type"),
              (axes[1], by_weekday, "DayOfWeek", lambda v: WEEKDAYS[int(v)], "By weekday")]
    top = max(np.nanmax(df[["model_uplift_pct", "observed_uplift_pct"]].to_numpy()) for _, df, *_ in panels)
    for ax, df, col, name, title in panels:
        x = np.arange(len(df))
        ax.set_xlim(-0.6, len(df) - 0.4)
        ax.set_ylim(0, top * 1.22)
        _rounded_bars(ax, x, df["model_uplift_pct"].to_numpy(), width=0.42, color=t["blue"])
        ax.scatter(x, df["observed_uplift_pct"], s=46, color=t["actual"], edgecolor=t["surface"],
                   linewidth=1.6, zorder=3)
        for xi, m, o in zip(x, df["model_uplift_pct"], df["observed_uplift_pct"]):
            ax.text(xi, max(m, o if np.isfinite(o) else m) + top * 0.035, f"+{m:.0f}%", ha="center", va="bottom",
                    fontsize=9, color=t["ink"])
        ax.set_xticks(x, [name(v) for v in df[col]])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}%"))
        ax.set_title(title, fontsize=9.5, color=t["ink2"], loc="left", pad=8)
    _titles(fig, t, "Extra sales on promotion days",
            "Mon–Fri store-days of the last 6 weeks, each forecast with and without the promotion.", top=0.96)
    handles = [plt.Rectangle((0, 0), 1, 1, color=t["blue"]),
               Line2D([], [], marker="o", ls="", markersize=7, color=t["actual"], markeredgecolor=t["surface"])]
    _legend(fig, t, handles, ["Model estimate", "Plain difference between promo and non-promo days"], y=0.845)
    _save(fig, out_dir, "promo_uplift", theme)


def feature_importance(importance: pd.DataFrame, out_dir: Path, theme: str, top_n: int = 12) -> None:
    t = THEMES[theme]
    top = importance.head(top_n).iloc[::-1]
    fig, ax = _new_figure(t, 10, 4.6)
    fig.subplots_adjust(left=0.25, right=0.95, top=0.80, bottom=0.05)
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", labelbottom=False)
    y = np.arange(len(top))
    share = 100 * top["gain_share"].to_numpy()
    ax.set_ylim(-0.6, len(top) - 0.4)
    ax.set_xlim(0, share.max() * 1.12)
    ax.axvline(0, color=t["axis"], lw=0.8)
    _rounded_bars(ax, y, share, width=0.5, color=t["blue"], horizontal=True)
    for yi, v in zip(y, share):
        ax.text(v + share.max() * 0.012, yi, f"{v:.1f}%", va="center", fontsize=9, color=t["ink"])
    ax.set_yticks(y, [FEATURE_LABELS.get(f, f) for f in top["feature"]])
    _titles(fig, t, "What the model relies on most",
            f"Share of the model's total gain, top {top_n} of {len(importance)} features.", top=0.965)
    _save(fig, out_dir, "feature_importance", theme)


def make_all(hold: pd.DataFrame, by_type: pd.DataFrame, by_weekday: pd.DataFrame,
             importance: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        forecast_total(hold, out_dir, theme)
        forecast_stores(hold, out_dir, theme)
        promo_uplift(by_type, by_weekday, out_dir, theme)
        feature_importance(importance, out_dir, theme)
    return sorted(out_dir.glob("*.png"))
