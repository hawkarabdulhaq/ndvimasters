# pages/02_Outliers_NDVI.py
# Streamlit page: Boxplot & Outliers (with Transform)
# Addresses questions 1,2,3,4,6 with dedicated tabs and topic-aligned maps.

import os
import io
import numpy as np
import streamlit as st
import rasterio
import plotly.express as px
import plotly.graph_objects as go

# ---------- CONFIG ----------
DEFAULT_NDVI_PATH = r"C:\Users\habdulhaq\Desktop\python\class_geomath_msc\ndvi_Algyo.tif"
DOWNSAMPLE_MAX = 900  # max width/height for on-screen images

# Optional SciPy for clustering/density; gracefully degrade if not available
try:
    from scipy import ndimage as ndi  # uniform_filter, label
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ---------- UTILS ----------
def _downsample(arr2d: np.ndarray, mask2d: np.ndarray, maxdim: int = DOWNSAMPLE_MAX):
    """Downsample array+mask for faster display."""
    h, w = arr2d.shape
    md = max(h, w)
    if md <= 0 or md <= maxdim:
        return arr2d, mask2d, 1
    scale = maxdim / md
    step = max(1, int(1 / scale))
    return arr2d[::step, ::step], mask2d[::step, ::step], step


@st.cache_data(show_spinner=False)
def load_ndvi_full(path: str):
    """
    Load NDVI from a single-band GeoTIFF.
    Returns:
      - flat 1D array of valid values
      - 2D array (original shape)
      - 2D boolean mask of valid pixels
      - geotransform meta (for future export if needed)
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        meta = src.meta.copy()

    mask = np.ones_like(arr, dtype=bool)
    if nodata is not None:
        mask &= arr != nodata
    mask &= ~np.isnan(arr)

    values = arr[mask]
    return values, arr, mask, meta


def fit_transform(arr2d: np.ndarray, mask2d: np.ndarray, method: str):
    """
    Apply transformation to NDVI for outlier detection.

    method:
      - 'raw': no transform (use NDVI directly)
      - 'log_shifted': robust shift using 2nd percentile, then log1p

    Returns:
      trans2d (same shape as arr2d), info dict with params
    """
    valid = arr2d[mask2d]
    if valid.size == 0:
        return np.zeros_like(arr2d), {"method": method}

    if method == "log_shifted":
        p2 = float(np.percentile(valid, 2))
        eps = 1e-6
        shifted = arr2d - p2 + eps
        shifted[~mask2d] = np.nan
        shifted = np.clip(shifted, a_min=eps, a_max=None)
        trans2d = np.log1p(shifted)
        return trans2d, {"method": method, "shift": p2, "eps": eps}
    else:
        # raw
        trans2d = arr2d.astype("float32").copy()
        trans2d[~mask2d] = np.nan
        return trans2d, {"method": "raw"}


def compute_box_stats(values: np.ndarray):
    """Compute quartiles, IQR, etc., ignoring NaNs."""
    v = values[~np.isnan(values)]
    if v.size == 0:
        return None
    q1 = float(np.percentile(v, 25))
    q3 = float(np.percentile(v, 75))
    iqr = float(q3 - q1)
    med = float(np.median(v))
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    return {"q1": q1, "q3": q3, "iqr": iqr, "median": med, "mean": mean, "std": std, "min": vmin, "max": vmax}


def outlier_thresholds(stats: dict, whisker_k: float):
    """Low/high cuts based on Q1 - k*IQR and Q3 + k*IQR."""
    low_cut = stats["q1"] - whisker_k * stats["iqr"]
    high_cut = stats["q3"] + whisker_k * stats["iqr"]
    return float(low_cut), float(high_cut)


def outlier_masks_from_transformed(trans2d: np.ndarray, mask2d: np.ndarray, low_cut: float, high_cut: float):
    """Return boolean masks for low/high outliers (computed in transformed domain)."""
    low_mask = (trans2d < low_cut) & mask2d
    high_mask = (trans2d > high_cut) & mask2d
    return low_mask, high_mask


def greyscale_from_ndvi(arr2d: np.ndarray, mask2d: np.ndarray):
    """Base greyscale NDVI image for maps."""
    arr2d_ds, mask2d_ds, _ = _downsample(arr2d, mask2d)
    h, w = arr2d_ds.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    valid = arr2d_ds[mask2d_ds]
    if valid.size == 0:
        rgb[~mask2d_ds] = (30, 30, 30)
        return rgb
    vmin = np.percentile(valid, 2)
    vmax = np.percentile(valid, 98)
    if vmin == vmax:
        vmin, vmax = np.min(valid), np.max(valid)
    norm = np.zeros_like(arr2d_ds, dtype="float32") if vmin == vmax else (arr2d_ds - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)
    base = (norm * 175 + 30).astype("uint8")  # retain contrast
    rgb[mask2d_ds] = np.stack([base, base, base], axis=-1)[mask2d_ds]
    rgb[~mask2d_ds] = (30, 30, 30)
    return rgb


def overlay_outliers(base_rgb: np.ndarray, arr2d: np.ndarray, mask2d: np.ndarray,
                     low_mask: np.ndarray, high_mask: np.ndarray, alpha: float = 0.85):
    """Overlay low/high outliers on base greyscale (low=red, high=green)."""
    # Downsample masks consistently for display
    arr2d_ds, mask2d_ds, step = _downsample(arr2d, mask2d)
    low_ds = low_mask[::step, ::step]
    high_ds = high_mask[::step, ::step]

    out = base_rgb.copy()
    # colors
    low_color = np.array([220, 60, 60], dtype=np.float32)
    high_color = np.array([60, 200, 80], dtype=np.float32)

    # apply overlay
    li = np.where(low_ds & mask2d_ds)
    hi = np.where(high_ds & mask2d_ds)
    for rr, cc in zip(li[0], li[1]):
        out[rr, cc] = (alpha * low_color + (1 - alpha) * out[rr, cc]).astype("uint8")
    for rr, cc in zip(hi[0], hi[1]):
        out[rr, cc] = (alpha * high_color + (1 - alpha) * out[rr, cc]).astype("uint8")
    return out


def boxplot_figure(values: np.ndarray, title: str, low_cut: float, high_cut: float, xlabel: str):
    """Plotly boxplot (in transformed domain if needed) + vertical lines for low/high cuts."""
    v = values[~np.isnan(values)]
    fig = px.box(v, points="outliers", title=title, labels={"value": xlabel})
    fig.add_vline(x=low_cut, line_width=2, line_dash="dash", line_color="red")
    fig.add_vline(x=high_cut, line_width=2, line_dash="dash", line_color="green")
    fig.update_layout(margin=dict(l=10, r=10, b=10, t=40))
    return fig


def label_components(mask: np.ndarray):
    """Connected component labeling (8-connectivity) if SciPy is available."""
    if not SCIPY_OK:
        return None, 0
    structure = np.ones((3, 3), dtype=int)
    labeled, ncomp = ndi.label(mask.astype(int), structure=structure)
    return labeled, ncomp


def density_heatmap(mask: np.ndarray, window: int = 25):
    """
    Local outlier density in [0,1]. Uses uniform_filter if SciPy is present.
    Fallback: block mean at coarse resolution.
    """
    if not SCIPY_OK or window < 3:
        # Fallback: block mean using coarse tiling
        h, w = mask.shape
        by = max(1, h // 200)
        bx = max(1, w // 200)
        H = (h // by) * by
        W = (w // bx) * bx
        small = mask[:H, :W].reshape(H // by, by, W // bx, bx).mean(axis=(1, 3))
        # upsample nearest
        heat = np.kron(small, np.ones((by, bx)))
        # pad back if needed
        out = np.zeros_like(mask, dtype=float)
        out[:heat.shape[0], :heat.shape[1]] = heat
        return np.clip(out, 0, 1)

    # SciPy path
    m = mask.astype(float)
    k = max(3, int(window))
    # two passes to compute mean in window
    num = ndi.uniform_filter(m, size=k, mode="nearest")
    den = ndi.uniform_filter(np.ones_like(m), size=k, mode="nearest")
    density = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return np.clip(density, 0, 1)


def heat_to_rgb(heat: np.ndarray, mask: np.ndarray):
    """
    Convert [0,1] heatmap to RGB (dark blue -> cyan/white).
    """
    heat_ds, mask_ds, _ = _downsample(heat, mask)
    h, w = heat_ds.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # simple blue->cyan/white scale
    rgb[..., 2] = (heat_ds * 255).astype("uint8")          # blue
    rgb[..., 1] = (heat_ds * 220).astype("uint8")          # green
    # light background for context
    rgb[~mask_ds] = (30, 30, 30)
    return rgb


def kmeans1d(values: np.ndarray, K: int = 2, max_iter: int = 50):
    """
    Simple 1D k-means (no external deps). Returns centers and labels for 1D array 'values'.
    NaNs ignored for center computation; labels for NaN entries set to -1.
    """
    x = values.copy()
    mask = ~np.isnan(x)
    xv = x[mask]
    if xv.size == 0:
        return np.array([]), np.full_like(x, -1, dtype=int)

    # init centers using percentiles
    centers = np.percentile(xv, np.linspace(10, 90, K))
    for _ in range(max_iter):
        # assign
        d = np.abs(xv[:, None] - centers[None, :])
        lab = np.argmin(d, axis=1)
        # update
        new_centers = np.array([np.mean(xv[lab == k]) if np.any(lab == k) else centers[k] for k in range(K)])
        if np.allclose(new_centers, centers, rtol=1e-4, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers

    # full label vector
    labels = np.full_like(x, -1, dtype=int)
    labels[mask] = lab
    return centers, labels


def recolor_labels_map(arr2d: np.ndarray, mask2d: np.ndarray, labels2d: np.ndarray):
    """Color clusters/components from 1D labels mapped to 2D."""
    # downsample consistently
    arr2d_ds, mask2d_ds, step = _downsample(arr2d, mask2d)
    labels_ds = labels2d[::step, ::step]
    h, w = arr2d_ds.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # palette (up to 4 clusters)
    palette = np.array(
        [
            [45, 160, 60],   # green
            [230, 210, 60],  # yellow
            [140, 70, 160],  # purple
            [70, 150, 220],  # blue
        ],
        dtype=np.uint8,
    )
    # base grey
    rgb[mask2d_ds] = (80, 80, 80)
    for k in range(4):
        where = (labels_ds == k) & mask2d_ds
        rgb[where] = palette[k % len(palette)]
    rgb[~mask2d_ds] = (30, 30, 30)
    return rgb


def to_png_download(rgb: np.ndarray, filename: str = "mask.png"):
    """Encode an RGB numpy array as PNG bytes for download."""
    from PIL import Image  # Pillow is usually available with Streamlit
    im = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue(), filename


# ---------- STREAMLIT PAGE ----------
def main():
    st.set_page_config(page_title="02 · NDVI Outliers", page_icon="🧪", layout="wide")
    st.title("02 · Boxplot & Outliers (with Transform): Maps & Reasoning")

    st.write(
        """
This page explores **outliers** in NDVI and uses them to answer five real‑life questions.
Use the sidebar to choose a **transform** and **whisker multiplier** for the boxplot rule.
Each tab shows a **map tailored to the question**.
"""
    )

    # --- Sidebar controls ---
    st.sidebar.header("NDVI Source")
    use_default = os.path.exists(DEFAULT_NDVI_PATH) and st.sidebar.checkbox(
        "Use default NDVI", value=True, help=DEFAULT_NDVI_PATH
    )
    uploaded = st.sidebar.file_uploader("…or upload NDVI GeoTIFF", type=["tif", "tiff"])

    if use_default:
        ndvi_path = DEFAULT_NDVI_PATH
        st.sidebar.success("Using default NDVI raster.")
    elif uploaded is not None:
        tmp_path = uploaded.name
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        ndvi_path = tmp_path
        st.sidebar.success(f"Using uploaded file: {uploaded.name}")
    else:
        st.warning("Provide an NDVI file (keep default path valid or upload in the sidebar).")
        return

    st.sidebar.header("Outlier Settings")
    transform = st.sidebar.radio(
        "Transform for outlier detection",
        options=["raw", "log_shifted"],
        index=0,
        help="Use 'log_shifted' to reduce skewness: shift via 2nd percentile and apply log1p.",
    )
    whisker_k = st.sidebar.slider(
        "IQR whisker multiplier (k)",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="Lower k flags more outliers; 1.5 is the classic Tukey rule.",
    )

    # Additional controls
    st.sidebar.header("Advanced")
    min_patch_size = st.sidebar.slider(
        "Coherent patch size threshold (pixels)",  # for Tab B
        min_value=1,
        max_value=500,
        value=40,
        step=1,
        help="Outlier clusters ≥ this size are treated as coherent features.",
    )
    density_win = st.sidebar.slider(
        "Local density window (px)",  # for Tab C
        min_value=5,
        max_value=101,
        value=35,
        step=2,
        help="Averaging window to estimate local outlier density (higher = smoother).",
    )
    gmm_k = st.sidebar.slider(
        "Number of groups for population-like segmentation (K)",  # for Tab D
        min_value=2, max_value=3, value=2, step=1
    )

    # --- Load data ---
    try:
        values_raw, ndvi2d, mask2d, meta = load_ndvi_full(ndvi_path)
    except Exception as e:
        st.error(f"Error loading NDVI raster: {e}")
        return

    if values_raw.size == 0:
        st.error("No valid NDVI values found.")
        return

    # --- Transform & Outlier thresholds (computed in transformed domain) ---
    trans2d, trans_info = fit_transform(ndvi2d, mask2d, transform)
    values_trans = trans2d[mask2d]
    stats_trans = compute_box_stats(values_trans)
    if stats_trans is None:
        st.error("Could not compute statistics on the selected data/transform.")
        return
    low_cut, high_cut = outlier_thresholds(stats_trans, whisker_k)
    low_mask, high_mask = outlier_masks_from_transformed(trans2d, mask2d, low_cut, high_cut)

    # For maps
    base_grey = greyscale_from_ndvi(ndvi2d, mask2d)
    overlay_map = overlay_outliers(base_grey, ndvi2d, mask2d, low_mask, high_mask, alpha=0.88)

    # Summary counts
    total_n = int(mask2d.sum())
    low_n = int(low_mask.sum())
    high_n = int(high_mask.sum())
    out_n = low_n + high_n
    low_pct = (100.0 * low_n / total_n) if total_n > 0 else 0.0
    high_pct = (100.0 * high_n / total_n) if total_n > 0 else 0.0

    # ---------- TABS ----------
    tabA, tabB, tabC, tabD, tabE = st.tabs(
        [
            "A · Unusually Low/High Vegetation",
            "B · Real Feature or Data Problem?",
            "C · Stability & Fragmentation",
            "D · Different Population?",
            "E · Anomalies & Quick Export",
        ]
    )

    # === TAB A ===
    with tabA:
        st.subheader("A · Are there parts of the town with unusually low/high vegetation?")
        colL, colR = st.columns([2, 1])

        with colL:
            st.image(
                overlay_map,
                caption="Red = low outliers, Green = high outliers (boxplot rule in transformed domain).",
                use_container_width=True,
            )

        with colR:
            st.write("**Boxplot (transformed domain)**")
            fig_box = boxplot_figure(
                values_trans,
                title="Boxplot with Outlier Thresholds",
                low_cut=low_cut,
                high_cut=high_cut,
                xlabel="Transformed NDVI" if transform != "raw" else "NDVI",
            )
            st.plotly_chart(fig_box, use_container_width=True)

            st.markdown(
                f"""
**Counts (k = {whisker_k:.1f}, transform = `{transform}`)**  
- Low outliers: **{low_n:,}**  ({low_pct:.2f}% of valid pixels)  
- High outliers: **{high_n:,}** ({high_pct:.2f}%)  
- Total outliers: **{out_n:,}**  ({(low_pct+high_pct):.2f}%)

**Interpretation:**  
- **Red clusters** → unusually **low NDVI** (barren/urban/water/shadow or artefacts).  
- **Green clusters** → unusually **high NDVI** (parks/forests/crops or artefacts).  
Use the **whisker multiplier** and **transform** to test robustness of what counts as “unusual”.
"""
            )

    # === TAB B ===
    with tabB:
        st.subheader("B · Do extreme values represent real features or data-quality problems?")
        colL, colR = st.columns([2, 1])

        # Component labeling and size-based coloring
        if SCIPY_OK:
            combined_out = (low_mask | high_mask) & mask2d
            labeled, ncomp = label_components(combined_out)
            # compute areas
            sizes = np.bincount(labeled.ravel())[1:]  # skip 0 (background)
            # Build a size-colored overlay
            arr2d_ds, mask2d_ds, step = _downsample(ndvi2d, mask2d)
            labeled_ds = labeled[::step, ::step] if labeled is not None else None
            base = base_grey.copy()
            overlay = base.copy()

            if labeled_ds is not None:
                # color small vs large components
                # small: pink, medium: orange, large: red
                # thresholds based on min_patch_size and 5x that
                small_thr = max(1, min_patch_size)
                large_thr = small_thr * 5
                # build a size lookup for downsampled labels
                import numpy as _np
                sizes_ds = _np.bincount(labeled.ravel())[1:]  # still original scale
                # map labels to colors
                for lab_id in range(1, ncomp + 1):
                    size = sizes_ds[lab_id - 1]
                    if size < small_thr:
                        color = (220, 140, 200)  # pinkish (likely noise)
                    elif size < large_thr:
                        color = (255, 160, 70)   # orange (possible small feature)
                    else:
                        color = (230, 60, 60)    # red (coherent feature)
                    mask_lab = (labeled_ds == lab_id) & mask2d_ds
                    overlay[mask_lab] = (0.85 * np.array(color) + 0.15 * overlay[mask_lab]).astype("uint8")
        else:
            overlay = overlay_map
            sizes = np.array([])
            ncomp = 0

        with colL:
            st.image(
                overlay,
                caption=(
                    "Outlier clusters colored by size: pink (tiny, often noise), orange (moderate), red (large, coherent).  "
                    "Adjust the 'Coherent patch size' in the sidebar."
                ),
                use_container_width=True,
            )

        with colR:
            if SCIPY_OK:
                n_small = int((sizes < min_patch_size).sum())
                n_large = int((sizes >= min_patch_size).sum())
                st.write(f"**Detected outlier clusters:** {int(len(sizes)):,}")
                st.write(f"- Small (< {min_patch_size} px): **{n_small:,}**")
                st.write(f"- Coherent (≥ {min_patch_size} px): **{n_large:,}**")
            else:
                st.warning("Install `scipy` to enable cluster sizing and coherence analysis.")

            st.markdown(
                """
**Reasoning:**  
- **Coherent, spatially contiguous clusters** (large red patches) are often **real land features** (river, park, fields).  
- **Isolated singletons** (pink speckles) are often **noise/artefacts** (shadows, sensor noise).  
Use this to decide: **clean the data** or **keep outliers** as meaningful targets.
"""
            )

    # === TAB C ===
    with tabC:
        st.subheader("C · Is vegetation stable or fragmented?")
        colL, colR = st.columns([2, 1])

        # local density of outliers
        combined_out = (low_mask | high_mask) & mask2d
        density = density_heatmap(combined_out, window=int(density_win))
        density_rgb = heat_to_rgb(density, mask2d)

        with colL:
            st.image(
                density_rgb,
                caption=f"Local outlier density (window ≈ {density_win} px). Brighter = more outliers nearby.",
                use_container_width=True,
            )

        with colR:
            frac_out = float(combined_out.sum()) / float(mask2d.sum())
            st.metric("Overall outlier fraction", f"{100.0*frac_out:.2f}%")
            st.write(
                """
**Interpretation:**  
- **High-density bands or patches** of outliers → **fragmentation** / sharp land-cover contrasts.  
- **Low, uniform density** → **stable** vegetation conditions.  
Compare this map with aerial imagery (if available) to confirm fragmentation corridors (roads, rivers, new builds).
"""
            )

        # Small boxplot for reference
        fig_ref = boxplot_figure(
            values_trans,
            title="Reference Boxplot (for context)",
            low_cut=low_cut,
            high_cut=high_cut,
            xlabel="Transformed NDVI" if transform != "raw" else "NDVI",
        )
        st.plotly_chart(fig_ref, use_container_width=True)

    # === TAB D ===
    with tabD:
        st.subheader("D · Do some areas behave like a different statistical population?")
        colL, colR = st.columns([2, 1])

        # 1D k-means on transformed values
        centers, labels_1d = kmeans1d(values_trans, K=int(gmm_k))
        # map back to 2D
        labels2d = np.full_like(ndvi2d, -1, dtype=int)
        labels2d[mask2d] = labels_1d
        pop_map = recolor_labels_map(ndvi2d, mask2d, labels2d)

        with colL:
            st.image(
                pop_map,
                caption=f"{gmm_k}-group segmentation in transformed domain: areas with distinct NDVI regimes.",
                use_container_width=True,
            )

        with colR:
            st.write("**Group centers (transformed values):**")
            if centers.size > 0:
                for i, c in enumerate(centers):
                    st.write(f"- Group {i}: center ≈ `{c:.4f}`")
            st.markdown(
                """
**Interpretation:**  
If one group aligns with **low NDVI** zones (industrial, riverbanks) and another with **high NDVI** zones (parks, fields),
you’re likely observing **different generating processes**.  
This supports modeling the town as a **mixture** rather than a single population.
"""
            )

        # Small histogram with centers
        v = values_trans[~np.isnan(values_trans)]
        fig_hist = px.histogram(v, nbins=50, title="Transformed NDVI Histogram with Group Centers")
        for c in centers:
            fig_hist.add_vline(x=float(c), line_dash="dot", line_color="white")
        st.plotly_chart(fig_hist, use_container_width=True)

    # === TAB E ===
    with tabE:
        st.subheader("E · Environmental anomalies & quick export")
        colL, colR = st.columns([2, 1])

        with colL:
            st.image(
                overlay_map,
                caption="Outlier hotspots (red/green) may indicate new builds, clearing, irrigation, or storm damage.",
                use_container_width=True,
            )

        with colR:
            st.write(
                """
**Real-life checks:**  
- **Unexpected green islands** → possible irrigation/crop emergence or encroachment.  
- **Unexpected low-NDVI scars** → clearing, construction, post-storm damage, or burn scars.  
- This is **single-date anomaly flagging**; true change detection needs a **time series**.
"""
            )
            # Provide PNG download of combined outlier mask overlay for quick sharing
            # (not georeferenced; for a georeferenced export, we'd write a GeoTIFF)
            try:
                # Build a simple red/green mask image for download
                down = overlay_map
                data, fname = to_png_download(down, filename="ndvi_outliers_overlay.png")
                st.download_button("⬇️ Download outlier overlay (PNG)", data=data, file_name=fname, mime="image/png")
            except Exception:
                st.warning("PNG export unavailable (missing Pillow).")

        # Also show thresholds & settings summary
        st.markdown(
            f"""
**Your settings recap:**  
- Transform: `{transform}` (info: {trans_info})  
- Whisker multiplier k: `{whisker_k:.2f}`  
- Low cut (transformed): `{low_cut:.4f}`  
- High cut (transformed): `{high_cut:.4f}`
"""
        )

    st.markdown("---")
    st.caption(
        "Boxplot rule uses Q1−k·IQR / Q3+k·IQR computed **in the chosen transform**. "
        "Changing transform or k lets you test whether outliers are robust or artefacts of skewness."
    )


if __name__ == "__main__":
    main()
