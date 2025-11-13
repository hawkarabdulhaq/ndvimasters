# pages/01_Distribution_NDVI.py

import os
import numpy as np
import streamlit as st
import rasterio
import plotly.express as px

# ---------- CONFIG ----------
DEFAULT_NDVI_PATH = r"C:\Users\habdulhaq\Desktop\python\class_geomath_msc\ndvi_Algyo.tif"
DOWNSAMPLE_MAX = 800  # max dimension for displayed images


# ---------- HELPERS ----------
def _downsample(arr2d: np.ndarray, mask2d: np.ndarray, downsample_max: int = DOWNSAMPLE_MAX):
    """Downsample array+mask for faster display."""
    h, w = arr2d.shape
    max_dim = max(h, w)
    if max_dim <= 0:
        return arr2d, mask2d
    if max_dim <= downsample_max:
        return arr2d, mask2d

    scale = downsample_max / max_dim
    step = int(1 / scale)
    if step < 1:
        step = 1
    arr2d_ds = arr2d[::step, ::step]
    mask2d_ds = mask2d[::step, ::step]
    return arr2d_ds, mask2d_ds


@st.cache_data(show_spinner=False)
def load_ndvi_full(path: str):
    """
    Load NDVI from a single-band GeoTIFF.
    Returns:
      - flat 1D array of valid values
      - 2D array (original shape)
      - 2D boolean mask of valid pixels
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata

    mask = np.ones_like(arr, dtype=bool)
    if nodata is not None:
        mask &= arr != nodata
    mask &= ~np.isnan(arr)

    values = arr[mask]
    return values, arr, mask


def compute_basic_stats(values: np.ndarray):
    """Compute key descriptive statistics for NDVI."""
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return {}

    mean = float(np.mean(clean))
    median = float(np.median(clean))
    q1 = float(np.percentile(clean, 25))
    q3 = float(np.percentile(clean, 75))
    iqr = float(q3 - q1)
    vmin = float(np.min(clean))
    vmax = float(np.max(clean))
    std = float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0

    counts, bin_edges = np.histogram(clean, bins=64)
    max_bin_idx = int(np.argmax(counts))
    mode_center = float((bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2)

    return {
        "n": int(clean.size),
        "mean": mean,
        "median": median,
        "mode_center": mode_center,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "std": std,
        "min": vmin,
        "max": vmax,
    }


def make_histogram(values: np.ndarray, bins: int = 40, title: str = "NDVI Distribution"):
    """Create a Plotly histogram figure for NDVI."""
    fig = px.histogram(
        x=values,
        nbins=bins,
        labels={"x": "NDVI value", "y": "Frequency"},
        title=title,
    )
    fig.update_layout(
        bargap=0.02,
        showlegend=False,
    )
    return fig


def make_ndvi_greenscale(arr2d: np.ndarray, mask2d: np.ndarray):
    """Map for A/E: basic NDVI green-scale (bright = high NDVI)."""
    arr2d, mask2d = _downsample(arr2d, mask2d)
    h, w = arr2d.shape
    valid = arr2d[mask2d]
    if valid.size == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)

    # robust stretch
    vmin = np.percentile(valid, 2)
    vmax = np.percentile(valid, 98)
    if vmin == vmax:
        vmin = np.min(valid)
        vmax = np.max(valid)
    if vmin == vmax:
        norm = np.zeros_like(arr2d, dtype="float32")
    else:
        norm = (arr2d - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 1] = (norm * 255).astype("uint8")  # green channel
    rgb[~mask2d] = (50, 50, 50)
    return rgb


def make_ndvi_class_map(arr2d: np.ndarray, mask2d: np.ndarray, q1: float, q3: float):
    """
    Map for B: classify NDVI into Low, Medium, High using Q1 & Q3.
    Low  < Q1
    Q1–Q3 = Medium
    > Q3 = High
    Colors: Low=purple-ish, Medium=yellow-ish, High=deep green.
    """
    arr2d, mask2d = _downsample(arr2d, mask2d)
    h, w = arr2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    low_mask = (arr2d < q1) & mask2d
    med_mask = (arr2d >= q1) & (arr2d <= q3) & mask2d
    high_mask = (arr2d > q3) & mask2d

    # base grey for all valid pixels
    rgb[mask2d] = (80, 80, 80)
    # low NDVI: purple
    rgb[low_mask] = (140, 70, 160)
    # medium NDVI: yellow
    rgb[med_mask] = (230, 220, 60)
    # high NDVI: green
    rgb[high_mask] = (40, 160, 40)

    # invalid = dark grey
    rgb[~mask2d] = (30, 30, 30)

    return rgb


def make_ndvi_variability_map(arr2d: np.ndarray, mask2d: np.ndarray, mean: float, std: float):
    """
    Map for C: shade pixels by |NDVI - mean| (absolute deviation).
    Larger deviation -> brighter color.
    """
    arr2d, mask2d = _downsample(arr2d, mask2d)
    h, w = arr2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    dev = np.abs(arr2d - mean)
    valid_dev = dev[mask2d]
    if valid_dev.size == 0:
        rgb[~mask2d] = (30, 30, 30)
        return rgb

    vmax = np.percentile(valid_dev, 98)
    if vmax <= 0:
        vmax = np.max(valid_dev)
    if vmax <= 0:
        # no variability at all
        norm = np.zeros_like(dev)
    else:
        norm = np.clip(dev / vmax, 0, 1)

    # use a blue-white gradient: small dev -> dark blue, large dev -> white
    blue = (norm * 255).astype("uint8")
    rgb[..., 2] = blue   # blue channel
    rgb[..., 1] = (norm * 200).astype("uint8")  # a bit of green, so high dev ~ cyan/white
    rgb[~mask2d] = (30, 30, 30)
    return rgb


def make_ndvi_extreme_map(arr2d: np.ndarray, mask2d: np.ndarray, q1: float, q3: float, iqr: float):
    """
    Map for D: highlight extremes outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    Base = grey NDVI; extremes = red.
    """
    arr2d, mask2d = _downsample(arr2d, mask2d)
    h, w = arr2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    low_cut = q1 - 1.5 * iqr
    high_cut = q3 + 1.5 * iqr

    extremes = ((arr2d < low_cut) | (arr2d > high_cut)) & mask2d

    # base greyscale for NDVI
    valid = arr2d[mask2d]
    if valid.size > 0:
        vmin = np.percentile(valid, 2)
        vmax = np.percentile(valid, 98)
        if vmin == vmax:
            vmin = np.min(valid)
            vmax = np.max(valid)
        if vmin == vmax:
            norm = np.zeros_like(arr2d, dtype="float32")
        else:
            norm = (arr2d - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0, 1)
        base = (norm * 180).astype("uint8")
        rgb[mask2d] = np.stack([base, base, base], axis=-1)[mask2d]

    # extremes in red
    rgb[extremes] = (230, 40, 40)

    # invalid
    rgb[~mask2d] = (30, 30, 30)

    return rgb


# ---------- STREAMLIT PAGE ----------
def main():
    st.set_page_config(page_title="01 · NDVI Distribution", page_icon="📊", layout="wide")

    st.title("01 · NDVI Distribution: Shape, Maps & Interpretation")

    st.write(
        """
This page uses the **distribution of NDVI** values for the town to answer
four key questions, each with a map tailored to the topic:

A. **What is the overall greenness level of the town?**  
B. **Does the town’s NDVI distribution show evidence of multiple land-cover types?**  
C. **How variable is vegetation across the town?**  
D. **Are there extreme NDVI values (very low or very high)? What do they imply?**  
E. **Extra tab for free exploration of map + histogram.**
"""
    )

    # --- Data source selection ---
    st.sidebar.header("NDVI Data Source")

    use_default = False
    if os.path.exists(DEFAULT_NDVI_PATH):
        use_default = st.sidebar.checkbox(
            "Use default NDVI raster", value=True,
            help=DEFAULT_NDVI_PATH
        )

    uploaded = st.sidebar.file_uploader("…or upload a GeoTIFF NDVI", type=["tif", "tiff"])

    if use_default and os.path.exists(DEFAULT_NDVI_PATH):
        ndvi_path = DEFAULT_NDVI_PATH
        st.sidebar.success("Using default NDVI raster.")
    elif uploaded is not None:
        tmp_path = uploaded.name
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        ndvi_path = tmp_path
        st.sidebar.success(f"Using uploaded file: {uploaded.name}")
    else:
        st.warning(
            "No NDVI file available. Either keep the default path valid on your machine, "
            "or upload a GeoTIFF NDVI raster in the sidebar."
        )
        return

    # --- Load data ---
    try:
        ndvi_values, ndvi_arr, ndvi_mask = load_ndvi_full(ndvi_path)
    except Exception as e:
        st.error(f"Error loading NDVI raster: {e}")
        return

    if ndvi_values.size == 0:
        st.error("No valid NDVI values found in the raster.")
        return

    stats = compute_basic_stats(ndvi_values)

    st.sidebar.subheader("Histogram Settings")
    bins = st.sidebar.slider(
        "Number of histogram bins",
        min_value=10,
        max_value=120,
        value=50,
        step=5,
    )

    # Precompute all map styles so tabs feel snappy
    map_overall = make_ndvi_greenscale(ndvi_arr, ndvi_mask)
    map_classes = make_ndvi_class_map(ndvi_arr, ndvi_mask, stats["q1"], stats["q3"])
    map_variab = make_ndvi_variability_map(ndvi_arr, ndvi_mask, stats["mean"], stats["std"])
    map_extreme = make_ndvi_extreme_map(ndvi_arr, ndvi_mask, stats["q1"], stats["q3"], stats["iqr"])

    # ---------- TABS ----------
    tabA, tabB, tabC, tabD, tabE = st.tabs(
        [
            "A · Overall Greenness",
            "B · Land-Cover Types",
            "C · Variability",
            "D · Extreme Values",
            "E · Explore Map & Histogram",
        ]
    )

    # --- TAB A: Overall greenness ---
    with tabA:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("NDVI Map – Overall Greenness")
            st.image(
                map_overall,
                caption="Green-scale NDVI: brighter = higher NDVI (more vegetation).",
                use_container_width=True,
            )

        with col2:
            st.subheader("Central Tendency of NDVI")
            st.metric("Number of valid pixels", f"{stats['n']:,}")
            c1, c2 = st.columns(2)
            c1.metric("Mean NDVI", f"{stats['mean']:.3f}")
            c2.metric("Median NDVI", f"{stats['median']:.3f}")
            st.metric("Mode (approx.)", f"{stats['mode_center']:.3f}")

            st.markdown(
                """
**How green is the town overall?**

- **Low values** (near 0 or negative) → mostly built-up / bare.  
- **Medium (≈ 0.3–0.5)** → mixed surfaces with some vegetation.  
- **High (> 0.5)** → vegetation dominates.

Compare mean, median and mode to decide if the NDVI is  
**mostly low, moderate, or high**, and whether the distribution is symmetric or skewed.
"""
            )

        st.markdown("### Histogram (context for greenness)")
        figA = make_histogram(ndvi_values, bins=bins, title="NDVI Distribution – Overall Greenness")
        st.plotly_chart(figA, use_container_width=True)

    # --- TAB B: Land-cover types ---
    with tabB:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("NDVI Map – Land-Cover Classes (Low / Medium / High)")
            st.image(
                map_classes,
                caption=(
                    "NDVI classes using Q1 & Q3: "
                    "Purple = low NDVI, Yellow = medium, Green = high."
                ),
                use_container_width=True,
            )

        with col2:
            st.subheader("Histogram – Shape & Modes")
            figB = make_histogram(ndvi_values, bins=bins, title="NDVI Distribution – Possible Land-Cover Modes")
            st.plotly_chart(figB, use_container_width=True)

        st.markdown(
            f"""
### Interpreting multiple land-cover types

We classify NDVI into three groups using quartiles:

- **Low**: NDVI < Q1 = `{stats['q1']:.3f}` (purple on the map)  
- **Medium**: Q1 ≤ NDVI ≤ Q3 (yellow)  
- **High**: NDVI > Q3 = `{stats['q3']:.3f}` (green)

**Histogram:**  
- One broad peak → town is dominated by a single land-cover intensity.  
- Two or more peaks → evidence for **multiple land-cover types**.

**Map:**  
- Clusters of purple → built-up / bare / water.  
- Clusters of green → vegetation (parks, fields).  

Combining both, you can argue whether the NDVI field is best described as a
**single process** or a **mixture of different land covers**.
"""
        )

    # --- TAB C: Variability ---
    with tabC:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("NDVI Deviation Map – Where Does NDVI Differ Most from the Mean?")
            st.image(
                map_variab,
                caption=(
                    "Brighter cyan/white ≈ NDVI far from the mean; "
                    "darker ≈ NDVI close to the mean."
                ),
                use_container_width=True,
            )

        with col2:
            st.subheader("Spread of NDVI Values")
            st.write(f"**Interquartile range (IQR)**: `{stats['iqr']:.3f}`")
            st.write(f"**Standard deviation**: `{stats['std']:.3f}`")
            st.write(f"**Q1 (25%)**: `{stats['q1']:.3f}`")
            st.write(f"**Q3 (75%)**: `{stats['q3']:.3f}`")

            st.markdown(
                """
**How variable is vegetation across the town?**

- **Small IQR & small standard deviation** → NDVI values are similar,
  the town is **fairly uniform** in greenness.  
- **Large IQR & large standard deviation** → NDVI values vary strongly,
  indicating a **patchy mosaic** of very green and very non-green areas.

The **deviation map** shows *where* NDVI differs most from the mean:
bright patches are **most different** (either much higher or much lower).
"""
            )

        st.markdown("### Histogram – Visualizing spread")
        figC = make_histogram(ndvi_values, bins=bins, title="NDVI Distribution – Variability")
        st.plotly_chart(figC, use_container_width=True)

    # --- TAB D: Extreme values ---
    with tabD:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("NDVI Extreme-Value Map")
            st.image(
                map_extreme,
                caption=(
                    "Red pixels = extreme NDVI values outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR]. "
                    "Grey = typical values."
                ),
                use_container_width=True,
            )

        with col2:
            st.subheader("Numerical Extremes")
            st.write(f"**Minimum NDVI**: `{stats['min']:.3f}`")
            st.write(f"**Maximum NDVI**: `{stats['max']:.3f}`")
            low_cut = stats["q1"] - 1.5 * stats["iqr"]
            high_cut = stats["q3"] + 1.5 * stats["iqr"]
            st.write(f"**Lower whisker cut (Q1 − 1.5·IQR)** ≈ `{low_cut:.3f}`")
            st.write(f"**Upper whisker cut (Q3 + 1.5·IQR)** ≈ `{high_cut:.3f}`")

            st.markdown(
                """
**What do the extremes represent?**

- Red pixels much **below** the bulk:  
  - water, shadows, asphalt, bare soil, or data errors.  
- Red pixels much **above** the bulk:  
  - dense forest, crops, parks, or noise if implausibly high.

The combination of:

1. **Min/max & whisker thresholds**,  
2. **Histogram tails**, and  
3. **Extreme-value map**

lets you decide whether these extremes are
**interesting geophysical features** or **outliers** that might be removed.
"""
            )

        st.markdown("### Histogram tails")
        figD = make_histogram(ndvi_values, bins=bins, title="NDVI Distribution – Tails & Extreme Values")
        st.plotly_chart(figD, use_container_width=True)

    # --- TAB E: Free exploration ---
    with tabE:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("NDVI Map – Raw Green-Scale")
            st.image(
                map_overall,
                caption="Raw NDVI visualization: green-scale.",
                use_container_width=True,
            )

        with col2:
            st.subheader("Histogram – Free Exploration")
            figE = make_histogram(ndvi_values, bins=bins, title="NDVI Distribution – Explore")
            st.plotly_chart(figE, use_container_width=True)

        st.markdown(
            """
Use this tab as a **sandbox**: play with the number of bins in the sidebar,
zoom in the histogram, and mentally connect **map patterns** with **distribution shape**.
"""
        )


if __name__ == "__main__":
    main()
