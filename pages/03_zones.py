# pages/3_zones.py
#
# Two-tab NDVI hypothesis testing with two *uploaded* rasters per tab.
# Tab 1: Parks (Zone 1) vs Residential (Zone 2)  -> default H1: mean(Zone1) > mean(Zone2)
# Tab 2: Industrial (Zone 1) vs Town (Zone 2)    -> default H1: mean(Zone1) < mean(Zone2)
#
# Each tab:
#   - Upload two NDVI GeoTIFFs (single-band).
#   - Shows comparable quick-look "maps" (common color scale).
#   - Extracts all valid NDVI pixels from each zone.
#   - Lets you choose Welch t-test or Mann–Whitney U; alpha; one/two-sided.
#   - Reports test statistic, p-value (if SciPy available), effect size (Cohen's d or Cliff's δ).
#   - Plots a compact two-group boxplot (sampling for speed only for the plot; stats use all pixels).

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
import plotly.express as px

# Optional SciPy for hypothesis tests
try:
    from scipy import stats as sstats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ---- CONFIG ----
MAX_IMAGE_DIM = 1024        # resize quick-look map images
PLOT_SAMPLE_MAX = 5000      # max points per group for plotting only (stats use ALL points)


# ---- NDVI I/O & QUICK-LOOKS ----
def load_ndvi(path: str):
    """Load a single-band NDVI GeoTIFF -> (ndvi2d, valid_mask, meta)."""
    with rasterio.open(path) as src:
        ndvi = src.read(1).astype("float32")
        nodata = src.nodata
        meta = src.meta.copy()
    valid = np.ones_like(ndvi, dtype=bool)
    if nodata is not None:
        valid &= ndvi != nodata
    valid &= ~np.isnan(ndvi)
    return ndvi, valid, meta


def robust_min_max_from_two(arr1, mask1, arr2, mask2, p_lo=2, p_hi=98):
    """Compute a common robust [min,max] across two NDVI sets for comparable rendering."""
    vals = np.concatenate([arr1[mask1], arr2[mask2]])
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(vals, p_lo))
    hi = float(np.percentile(vals, p_hi))
    if lo == hi:
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if lo == hi:
            lo, hi = 0.0, 1.0
    return lo, hi


def ndvi_to_greenscale(ndvi2d, mask2d, lo, hi, maxdim=MAX_IMAGE_DIM):
    """Render NDVI to a green-scale PNG-like RGB np.array with given stretch [lo, hi]."""
    h, w = ndvi2d.shape
    # downsample factor
    factor = max(1, int(max(h, w) // maxdim)) if max(h, w) > maxdim else 1
    ndvi_ds = ndvi2d[::factor, ::factor]
    mask_ds = mask2d[::factor, ::factor]
    rgb = np.zeros((ndvi_ds.shape[0], ndvi_ds.shape[1], 3), dtype=np.uint8)
    if lo == hi:
        norm = np.zeros_like(ndvi_ds, dtype="float32")
    else:
        norm = (ndvi_ds - lo) / (hi - lo)
    norm = np.clip(norm, 0, 1)
    rgb[..., 1] = (norm * 255).astype("uint8")      # green channel
    rgb[~mask_ds] = (40, 40, 40)                    # dark grey for invalid
    return rgb


def bounds_wgs84(meta):
    """Return raster bounds as ((south, west), (north, east)) in EPSG:4326 (for info)."""
    h, w = meta["height"], meta["width"]
    left, bottom, right, top = array_bounds(h, w, meta["transform"])
    b = transform_bounds(meta["crs"], "EPSG:4326", left, bottom, right, top, densify_pts=21)
    return (b[1], b[0]), (b[3], b[2])  # (south, west), (north, east)


# ---- TESTS & EFFECT SIZES ----
def welch_ttest(a, b, alt="two-sided"):
    """Welch t-test via SciPy if available."""
    res = {"test": "Welch t-test", "stat": np.nan, "p": np.nan}
    if SCIPY_OK:
        t = sstats.ttest_ind(a, b, equal_var=False, alternative=alt)
        res["stat"] = float(t.statistic)
        res["p"] = float(t.pvalue)
    else:
        # compute t only (no p without SciPy)
        n1, n2 = len(a), len(b)
        if n1 >= 2 and n2 >= 2:
            m1, m2 = np.mean(a), np.mean(b)
            v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
            res["stat"] = float((m1 - m2) / np.sqrt(v1 / n1 + v2 / n2 + 1e-12))
    return res


def mann_whitney(a, b, alt="two-sided"):
    """Mann–Whitney U (requires SciPy)."""
    res = {"test": "Mann–Whitney U", "stat": np.nan, "p": np.nan, "U": np.nan}
    if SCIPY_OK:
        u = sstats.mannwhitneyu(a, b, alternative=alt)
        res["stat"] = float(u.statistic)
        res["p"] = float(u.pvalue)
        res["U"] = float(u.statistic)
    return res


def cohens_d(a, b):
    """Cohen's d (pooled SD)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / max(na + nb - 2, 1))
    if sp == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / sp)


def cliffs_delta_from_U(U, n1, n2):
    """Cliff's δ from U: δ = 2U/(n1*n2) - 1."""
    if n1 == 0 or n2 == 0 or np.isnan(U):
        return np.nan
    return float((2.0 * U) / (n1 * n2) - 1.0)


# ---- PLOTS ----
def two_group_box(df, value_col="NDVI", group_col="Zone", title=""):
    fig = px.box(df, x=group_col, y=value_col, points="outliers", title=title,
                 labels={group_col: "", value_col: "NDVI"})
    fig.update_layout(margin=dict(l=10, r=10, b=10, t=50))
    return fig


# ---- UI HELPER (one tab) ----
def run_two_raster_test(
    tab_title: str,
    zone1_label: str,
    zone2_label: str,
    default_alt: str,
    default_test: str,
    map_key_prefix: str
):
    st.subheader(tab_title)

    # Uploaders (two rasters)
    c_up1, c_up2 = st.columns(2)
    f1 = c_up1.file_uploader(f"Upload NDVI GeoTIFF for **{zone1_label}** (Zone 1)", type=["tif", "tiff"], key=f"{map_key_prefix}_z1")
    f2 = c_up2.file_uploader(f"Upload NDVI GeoTIFF for **{zone2_label}** (Zone 2)", type=["tif", "tiff"], key=f"{map_key_prefix}_z2")

    # Test controls
    c1, c2, c3 = st.columns([1.1, 1.1, 1])
    test_type = c1.radio(
        "Test type", options=["Welch t-test", "Mann–Whitney U"],
        index=0 if default_test == "Welch t-test" else 1,
        help="Welch compares means (robust to unequal variances); Mann–Whitney is non-parametric (ranks/medians).",
        key=f"{map_key_prefix}_test_type"
    )
    alt_dir = c2.selectbox(
        "Alternative (H₁)", options=["two-sided", "greater", "less"],
        index={"two-sided": 0, "greater": 1, "less": 2}[default_alt],
        help="Choose one-sided if the question is directional (e.g., Parks > Residential).",
        key=f"{map_key_prefix}_alt_dir"
    )
    alpha = c3.slider("α (significance level)", 0.001, 0.10, 0.05, 0.001, key=f"{map_key_prefix}_alpha")

    # Need both rasters
    if not f1 or not f2:
        st.info("Upload **both** NDVI files to proceed.")
        return

    # Load both
    try:
        ndvi1, valid1, meta1 = load_ndvi(f1)
        ndvi2, valid2, meta2 = load_ndvi(f2)
    except Exception as e:
        st.error(f"Error reading one of the rasters: {e}")
        return

    # Common stretch for comparable rendering
    lo, hi = robust_min_max_from_two(ndvi1, valid1, ndvi2, valid2)
    rgb1 = ndvi_to_greenscale(ndvi1, valid1, lo, hi)
    rgb2 = ndvi_to_greenscale(ndvi2, valid2, lo, hi)

    # Show quick-look "maps"
    st.markdown("### NDVI quick-look (common color scale)")
    cm1, cm2 = st.columns(2)
    cm1.image(rgb1, caption=f"{zone1_label} — green = higher NDVI", use_container_width=True)
    cm2.image(rgb2, caption=f"{zone2_label} — green = higher NDVI", use_container_width=True)

    # Extract all valid values
    vals1 = ndvi1[valid1]
    vals2 = ndvi2[valid2]
    n1, n2 = len(vals1), len(vals2)
    if n1 == 0 or n2 == 0:
        st.warning("One of the rasters has no valid NDVI values.")
        return

    # Run chosen test
    if test_type == "Welch t-test":
        test_res = welch_ttest(vals1, vals2, alt=alt_dir)
        eff_size = cohens_d(vals1, vals2)  # Cohen's d
        eff_text = f"Cohen’s d = {eff_size:.3f} (|0.2| small, |0.5| medium, |0.8| large)"
        stat_lbl = "t"
    else:
        test_res = mann_whitney(vals1, vals2, alt=alt_dir)
        # Effect size via U if available
        if SCIPY_OK and not np.isnan(test_res.get("U", np.nan)):
            eff_size = cliffs_delta_from_U(test_res["U"], n1, n2)
            eff_text = f"Cliff’s δ = {eff_size:.3f} (|0.147| small, |0.33| medium, |0.474| large)"
        else:
            eff_text = "Cliff’s δ = N/A (install SciPy for U statistic)"
        stat_lbl = "U"

    # Summary metrics
    mean1, mean2 = float(np.mean(vals1)), float(np.mean(vals2))
    med1, med2 = float(np.median(vals1)), float(np.median(vals2))

    st.markdown("### Results")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("n (Zone 1)", f"{n1:,}")
    r2.metric("n (Zone 2)", f"{n2:,}")
    r3.metric(f"{stat_lbl} statistic", f"{test_res['stat']:.3f}" if not np.isnan(test_res["stat"]) else "N/A")
    r4.metric("p-value", f"{test_res['p']:.4g}" if not np.isnan(test_res["p"]) else "N/A")

    decision = "—"
    if not np.isnan(test_res["p"]):
        decision = f"Reject H₀ at α={alpha:.3f}." if test_res["p"] < alpha else f"Do not reject H₀ at α={alpha:.3f}."
    st.success(f"**Decision:** {decision}")
    st.caption(eff_text)

    st.markdown(
        f"""
**{zone1_label}**: mean = `{mean1:.3f}`, median = `{med1:.3f}`  
**{zone2_label}**: mean = `{mean2:.3f}`, median = `{med2:.3f}`
"""
    )

    # Boxplot (sample for plot only to keep it responsive)
    s1 = vals1[:PLOT_SAMPLE_MAX]
    s2 = vals2[:PLOT_SAMPLE_MAX]
    dfp = pd.DataFrame({
        "NDVI": np.concatenate([s1, s2]),
        "Zone": [zone1_label] * len(s1) + [zone2_label] * len(s2)
    })
    fig = two_group_box(dfp, title="NDVI by Zone (Boxplot with outliers)")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Notes: Type I error = false positive (reject a true H₀); Type II = false negative (retain a false H₀). "
        "With very large n, even tiny differences can be significant—use effect sizes to judge practical importance."
    )


# ---- PAGE ----
def main():
    st.set_page_config(page_title="03 · Hypothesis Tests with Two NDVI Files", page_icon="🧭", layout="wide")
    st.title("03 · Hypothesis Testing (Two Uploaded NDVI Zones)")

    st.write(
        """
Upload **two NDVI rasters** per tab (Zone 1 and Zone 2).  
We compare their NDVI distributions/statistics using either **Welch t‑test** (means) or **Mann–Whitney U** (non‑parametric).
The quick-look images use a **common color scale** for fair visual comparison.
"""
    )

    tab1, tab2 = st.tabs(["Parks vs Residential", "Industrial vs Town"])

    with tab1:
        st.markdown(
            "> **Question**: Are parks significantly greener than residential neighborhoods?\n"
            "Upload an NDVI for **Parks (Zone 1)** and an NDVI for **Residential (Zone 2)**."
        )
        run_two_raster_test(
            tab_title="Parks (Zone 1) vs Residential (Zone 2)",
            zone1_label="Parks",
            zone2_label="Residential",
            default_alt="greater",          # Parks > Residential
            default_test="Welch t-test",
            map_key_prefix="parks_resid",
        )

    with tab2:
        st.markdown(
            "> **Question**: Are industrial zones statistically less green than the rest of the town?\n"
            "Upload an NDVI for **Industrial (Zone 1)** and an NDVI for the **Town (Zone 2)**."
        )
        run_two_raster_test(
            tab_title="Industrial (Zone 1) vs Town (Zone 2)",
            zone1_label="Industrial",
            zone2_label="Town",
            default_alt="less",             # Industrial < Town
            default_test="Welch t-test",
            map_key_prefix="ind_town",
        )


if __name__ == "__main__":
    main()
