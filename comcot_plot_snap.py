#!/usr/bin/env python
"""COMCOT snapshot plotter.
First install dependencies:
pip install numpy matplotlib scipy imageio imageio-ffmpeg

Usage examples:
    # plot the latest snapshot for layer 01 (auto-detected)
    python comcot_plot_snap.py 01

    # plot a specific snapshot index with z prefix and verbose output
    python comcot_plot_snap.py 01 120 --surface-prefix z --verbose

    # plot a specific snapshot index with h prefix
    python comcot_plot_snap.py 01 120 --surface-prefix h

    # plot all available snapshots for layer 01 with timing and verbose output
    python comcot_plot_snap.py 01 --all --mp4 --outdir ../output/plots --fps 8 --verbose

    # build MP4 only from existing PNGs in input directory (no re-plot)
    python comcot_plot_snap.py 01 --input-dir ./existing_plots --outdir ./videos --mp4-only --fps 8

    # plot snapshots from a different base directory
    python comcot_plot_snap.py 01 --base /path/to/comcot/output --outdir ./plots

    # plot with custom stride for velocity arrows
    python comcot_plot_snap.py 01 120 --stride 5 --verbose

    # plot all snapshots with custom MP4 name
    python comcot_plot_snap.py 01 --all --mp4 --mp4-name tsunami_animation.mp4
"""

from pathlib import Path
import argparse
from typing import Optional, Sequence
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.colors import ListedColormap

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None


# Hardcoded sediment colormap (extracted from sediment_cmap.mat)
# This colormap goes from deep blue through cyan, white, yellow to red
SEDIMENT_CMAP = np.array([
    [0.000000, 0.000000, 1.000000],
    [0.016000, 0.032000, 1.000000],
    [0.032000, 0.064000, 1.000000],
    [0.048000, 0.096000, 1.000000],
    [0.064000, 0.128000, 1.000000],
    [0.080000, 0.160000, 1.000000],
    [0.096000, 0.192000, 1.000000],
    [0.112000, 0.224000, 1.000000],
    [0.128000, 0.256000, 1.000000],
    [0.144000, 0.288000, 1.000000],
    [0.160000, 0.320000, 1.000000],
    [0.176000, 0.352000, 1.000000],
    [0.192000, 0.384000, 1.000000],
    [0.208000, 0.416000, 1.000000],
    [0.224000, 0.448000, 1.000000],
    [0.240000, 0.480000, 1.000000],
    [0.256000, 0.512000, 1.000000],
    [0.272000, 0.544000, 1.000000],
    [0.288000, 0.576000, 1.000000],
    [0.304000, 0.608000, 1.000000],
    [0.320000, 0.640000, 1.000000],
    [0.336000, 0.672000, 1.000000],
    [0.352000, 0.704000, 1.000000],
    [0.368000, 0.736000, 1.000000],
    [0.384000, 0.768000, 1.000000],
    [0.400000, 0.800000, 1.000000],
    [0.485714, 0.828571, 1.000000],
    [0.571429, 0.857143, 1.000000],
    [0.657143, 0.885714, 1.000000],
    [0.742857, 0.914286, 1.000000],
    [0.828571, 0.942857, 1.000000],
    [0.914286, 0.971429, 1.000000],
    [1.000000, 1.000000, 1.000000],
    [1.000000, 1.000000, 0.800000],
    [1.000000, 1.000000, 0.600000],
    [1.000000, 1.000000, 0.400000],
    [1.000000, 1.000000, 0.200000],
    [1.000000, 1.000000, 0.000000],
    [1.000000, 0.961538, 0.000000],
    [1.000000, 0.923077, 0.000000],
    [1.000000, 0.884615, 0.000000],
    [1.000000, 0.846154, 0.000000],
    [1.000000, 0.807692, 0.000000],
    [1.000000, 0.769231, 0.000000],
    [1.000000, 0.730769, 0.000000],
    [1.000000, 0.692308, 0.000000],
    [1.000000, 0.653846, 0.000000],
    [1.000000, 0.615385, 0.000000],
    [1.000000, 0.576923, 0.000000],
    [1.000000, 0.538462, 0.000000],
    [1.000000, 0.500000, 0.000000],
    [1.000000, 0.461538, 0.000000],
    [1.000000, 0.423077, 0.000000],
    [1.000000, 0.384615, 0.000000],
    [1.000000, 0.346154, 0.000000],
    [1.000000, 0.307692, 0.000000],
    [1.000000, 0.269231, 0.000000],
    [1.000000, 0.230769, 0.000000],
    [1.000000, 0.192308, 0.000000],
    [1.000000, 0.153846, 0.000000],
    [1.000000, 0.115385, 0.000000],
    [1.000000, 0.076923, 0.000000],
    [1.000000, 0.038462, 0.000000],
    [1.000000, 0.000000, 0.000000],
])


SURFACE_PREFIX_CANDIDATES: Sequence[str] = ("z", "h", "eta", "snap")
VELOCITY_PREFIX_PAIRS: Sequence[tuple[str, str]] = (("m", "n"), ("u", "v"))


def contour_interval(mat: np.ndarray) -> int:
    vmin = float(np.nanmin(mat))
    if vmin > -500:
        return 10
    if vmin > -1000:
        return 50
    if vmin > -8000:
        return 100
    return 200


def load_colormap(mat_file: Path | None) -> str | ListedColormap:
    # Always use the hardcoded sediment colormap to match MATLAB output
    return ListedColormap(SEDIMENT_CMAP)


def load_grid(layer: str, base: Path) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    x = np.loadtxt(base / f"layer{layer}_x.dat")
    y = np.loadtxt(base / f"layer{layer}_y.dat")
    nx, ny = len(x), len(y)
    bathy_path = base / f"layer{layer}.dat"
    if bathy_path.exists():
        bathy = np.loadtxt(bathy_path)
        bathy = np.reshape(bathy, (nx, ny), order="F")  # match MATLAB column-major reshape
    else:
        bathy = None
    return x, y, bathy


def find_snapshot_indices(layer: str, base: Path, surface_prefix: Optional[str] = None) -> tuple[str, list[int]]:
    """Return (detected_surface_prefix, sorted_snapshot_indices)."""
    prefixes = [surface_prefix] if surface_prefix else list(SURFACE_PREFIX_CANDIDATES)
    for prefix in prefixes:
        if prefix is None:
            continue
        snaps: list[int] = []
        for path in base.glob(f"{prefix}_{layer}_*.dat"):
            tail = path.stem.split("_")[-1]
            if tail.isdigit():
                snaps.append(int(tail))
        if snaps:
            return prefix, sorted(snaps)
    return "", []


def load_flat(path: Path) -> np.ndarray:
    """Load whitespace-separated numbers even if rows have uneven column counts."""
    data = path.read_text()
    return np.fromstring(data, sep=" ", dtype=float)[:, None]  # column vector


def load_snapshot(layer: str, snap: int, base: Path, surface_prefix: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    suffix = f"{snap:06d}"
    surf_path = base / f"{surface_prefix}_{layer}_{suffix}.dat"
    if not surf_path.exists():
        # Return None for all arrays if file doesn't exist
        return None, None, None
    z = load_flat(surf_path)

    m_arr: Optional[np.ndarray] = None
    n_arr: Optional[np.ndarray] = None
    for vx, vy in VELOCITY_PREFIX_PAIRS:
        vx_path = base / f"{vx}_{layer}_{suffix}.dat"
        vy_path = base / f"{vy}_{layer}_{suffix}.dat"
        if vx_path.exists() and vy_path.exists():
            m_arr = load_flat(vx_path)
            n_arr = load_flat(vy_path)
            break

    return z, m_arr, n_arr


def reshape_to_grid(arr: Optional[np.ndarray], nx: int, ny: int) -> Optional[np.ndarray]:
    if arr is None:
        return None
    return np.reshape(arr, (nx, ny), order="F")


def frames_from_outdir(layer: str, outdir: Path) -> list[Path]:
    """Return sorted frames for this layer from outdir.
    Supports both JPG (snap_{layer}_*.jpg) and PNG (z_{layer}_*.png) patterns."""
    # Try PNG pattern first (z_{layer}_*.png)
    frames = sorted(outdir.glob(f"z_{layer}_*.png"))
    if not frames:
        # Fallback to JPG pattern (snap_{layer}_*.jpg)
        frames = sorted(outdir.glob(f"snap_{layer}_*.jpg"))
    return frames


def frames_from_input_dir(layer: str, input_dir: Path) -> list[Path]:
    """Return sorted frames for this layer from input directory.
    Supports both JPG (snap_{layer}_*.jpg) and PNG (z_{layer}_*.png) patterns."""
    # Try PNG pattern first (z_{layer}_*.png)
    frames = sorted(input_dir.glob(f"z_{layer}_*.png"))
    if not frames:
        # Fallback to JPG pattern (snap_{layer}_*.jpg)
        frames = sorted(input_dir.glob(f"snap_{layer}_*.jpg"))
    return frames


def write_mp4(frames: Sequence[Path], mp4_path: Path, fps: int) -> Path:
    if not frames:
        raise ValueError("No frames to write to MP4.")
    if imageio is None:
        raise RuntimeError("imageio is required for MP4 output. Install with `pip install imageio imageio-ffmpeg`.")
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(mp4_path, fps=fps) as writer:
        for frame in frames:
            # imageio.imread handles both PNG and JPG formats
            writer.append_data(imageio.imread(frame))
    return mp4_path


def plot_snapshot(
    x: np.ndarray,
    y: np.ndarray,
    bathy: Optional[np.ndarray],
    z: Optional[np.ndarray],
    m: Optional[np.ndarray],
    n: Optional[np.ndarray],
    layer: str,
    snap: int,
    surface_prefix: str,
    cmap: str | ListedColormap,
    stride: int,
    output: Path,
) -> Path:
    if z is None:
        raise ValueError("Surface elevation data (z) cannot be None")
    
    nx, ny = len(x), len(y)
    z_grid = reshape_to_grid(z, nx, ny)
    m_grid = reshape_to_grid(m, nx, ny)
    n_grid = reshape_to_grid(n, nx, ny)

    cmax = float(np.nanmax(z_grid)) / 2 if np.size(z_grid) else 0.0
    # Use fixed color range to match MATLAB code (caxis([-12 12]))
    vmin, vmax = -12, 12
    dc = contour_interval(bathy) if bathy is not None else None

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot surface elevation with sediment colormap
    mesh = ax.pcolormesh(x, y, z_grid.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = fig.colorbar(mesh, ax=ax, label="Surface elevation")
    
    # Add velocity vectors if available
    if m_grid is not None and n_grid is not None:
        step = slice(None, None, stride)
        ax.quiver(
            x[step],
            y[step],
            m_grid[step, step].T,
            n_grid[step, step].T,
            color="r",
            linewidth=0.7,
            scale_units="xy",
            scale=1,
        )

    # Add bathymetry contours
    if bathy is not None and dc is not None:
        # Negative contours (below sea level)
        levels = np.arange(-8000, 0, dc)
        ax.contour(x, y, bathy.T, levels=levels, colors=[(0.5, 0.5, 0.5)], linewidths=0.6, linestyles='--')
        # Coastline (zero contour)
        ax.contour(x, y, bathy.T, levels=[0], colors="k", linewidths=0.8, zorder=5)

    # Set plot properties
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    hours, remainder = divmod(snap, 3600)
    minutes, seconds = divmod(remainder, 60)
    timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    ax.set_title(f"Layer-{surface_prefix.upper()} {layer}, {timestamp} after eq")
    ax.set_axisbelow(False)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot COMCOT snapshots.")
    parser.add_argument("layer", help="Layer id, e.g. 01")
    parser.add_argument("snaptime", type=int, nargs="?", help="Snapshot time index (integer). If omitted, use latest.")
    parser.add_argument("--base", type=Path, default=Path("."), help="Directory with COMCOT outputs")
    parser.add_argument("--surface-prefix", type=str, help="Surface file prefix (z, h, eta, or snap). If not specified, auto-detects.")
    parser.add_argument("--cmap-mat", type=Path, default=Path("parulaz.mat"), help="MAT-file with mycmap (ignored, using hardcoded colormap)")
    parser.add_argument("--stride", type=int, default=10, help="Subsample step for quiver arrows")
    parser.add_argument("--outdir", type=Path, default=Path("."), help="Output directory for the figure(s) and MP4")
    parser.add_argument("--input-dir", type=Path, help="Input directory for existing frames (default: same as --outdir)")
    parser.add_argument("--all", action="store_true", help="Plot all snapshot indices found for this layer")
    parser.add_argument("--mp4", action="store_true", help="Write MP4 from generated JPG frames (requires imageio).")
    parser.add_argument("--mp4-name", type=str, help="Filename for MP4 (default: snap_<layer>.mp4 in outdir)")
    parser.add_argument("--fps", type=int, default=6, help="Frame rate for MP4 when --mp4 is set")
    parser.add_argument("--verbose", action="store_true", help="Display detailed information about file processing")
    parser.add_argument(
        "--mp4-only",
        action="store_true",
        help="Skip plotting and build MP4 from existing frames in input directory for this layer.",
    )
    args = parser.parse_args()

    if args.mp4_only:
        input_dir = args.input_dir if args.input_dir else args.outdir
        frames = frames_from_input_dir(args.layer, input_dir)
        if not frames:
            raise SystemExit(f"No existing frames found in {input_dir} matching z_{args.layer}_*.png or snap_{args.layer}_*.jpg")
        mp4_name = args.mp4_name if args.mp4_name else f"z_{args.layer}.mp4"
        mp4_path = args.outdir / mp4_name
        video = write_mp4(frames, mp4_path, args.fps)
        print(f"Saved {video}")
        return

    surface_prefix, snaps = find_snapshot_indices(args.layer, args.base, args.surface_prefix)
    if not snaps:
        raise SystemExit(
            f"No snapshot files found for layer {args.layer} in {args.base}. "
            f"Tried prefixes {([args.surface_prefix] if args.surface_prefix else list(SURFACE_PREFIX_CANDIDATES))}."
        )

    if args.all:
        target_snaps = snaps
    elif args.snaptime is None:
        target_snaps = [snaps[-1]]  # latest
    else:
        target_snaps = [args.snaptime]

    x, y, bathy = load_grid(args.layer, args.base)
    cmap = load_colormap(args.cmap_mat if args.cmap_mat else None)

    # Start timer
    start_time = time.time()
    processed_count = 0
    skipped_count = 0

    if args.verbose:
        print(f"Processing {len(target_snaps)} snapshot(s) for layer {args.layer}")
        print(f"Using surface prefix: '{surface_prefix}'")
        print(f"Output directory: {args.outdir}")
        print("-" * 50)

    saved_frames: list[Path] = []
    for snap in target_snaps:
        z, m, n = load_snapshot(args.layer, snap, args.base, surface_prefix)
        if z is None:
            skipped_count += 1
            if args.verbose:
                print(f"Skipping snapshot {snap:06d} - file not found")
            continue
        
        processed_count += 1
        if args.verbose:
            print(f"Processing snapshot {snap:06d}...", end=" ")
        
        suffix = f"{snap:06d}"
        outfile = args.outdir / f"z_{args.layer}_{suffix}.png"
        saved = plot_snapshot(
            x,
            y,
            bathy,
            z,
            m,
            n,
            args.layer,
            snap,
            surface_prefix,
            cmap,
            args.stride,
            outfile,
        )
        
        if args.verbose:
            print(f"Saved to {outfile.name}")
        else:
            print(f"Saved {saved}")
        saved_frames.append(saved)

    # End timer and report
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("-" * 50)
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Processed: {processed_count} snapshots")
    print(f"Skipped: {skipped_count} snapshots")
    if processed_count > 0:
        print(f"Average time per snapshot: {elapsed_time/processed_count:.2f} seconds")

    if args.mp4:
        frames_for_mp4 = saved_frames if saved_frames else frames_from_outdir(args.layer, args.outdir)
        if not frames_for_mp4:
            raise SystemExit(f"No frames available to build MP4 in {args.outdir} (expected z_{args.layer}_*.png or snap_{args.layer}_*.jpg)")
        mp4_name = args.mp4_name if args.mp4_name else f"z_{args.layer}.mp4"
        mp4_path = args.outdir / mp4_name
        video = write_mp4(frames_for_mp4, mp4_path, args.fps)
        print(f"Saved {video}")


if __name__ == "__main__":
    main()
