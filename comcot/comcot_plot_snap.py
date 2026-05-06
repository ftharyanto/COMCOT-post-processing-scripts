#!/usr/bin/env python
"""
COMCOT Snapshot Plotter

Dependencies:
    - numpy
    - matplotlib
    - scipy
    - imageio
    - imageio-ffmpeg (for MP4 support)

Install command:
    pip install numpy matplotlib scipy imageio imageio-ffmpeg

This script handles two main tasks:
1. Plotting: Converting raw COMCOT simulation data (.dat files) into maps (.png).
2. Video: Stitching images (.png/.jpg) into a video (.mp4).

Path Arguments:
    --dat-dir:    Where the raw simulation .dat files are located.
    --img-dir:    Where existing plot images (.png/.jpg) are located (for video-only tasks).
    --output-dir: Where new plots and videos will be saved.

Usage Examples:
    # 1. Plot the latest snapshot from the current directory (No arrows by default)
    python comcot_plot_snap.py 01

    # 2. Plot all snapshots and include velocity arrows
    python comcot_plot_snap.py 01 --all --arrows

    # 3. Plot all snapshots and create a video immediately
    python comcot_plot_snap.py 01 --all --mp4 --output-dir ./results

    # 4. Generate a video ONLY from existing images in a folder
    python comcot_plot_snap.py 01 --video-only --img-dir ./plots --output-dir ./videos --fps 10

    # 5. Plot with arrows and custom arrow density
    python comcot_plot_snap.py 01 --arrows --stride 5
"""

from pathlib import Path
import argparse
from typing import Optional, Sequence
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

# For Windows-specific creation time updates
try:
    import ctypes
    from ctypes import wintypes
except ImportError:
    ctypes = None

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

# Hardcoded sediment colormap
SEDIMENT_CMAP = np.array([
    [0.000000, 0.000000, 1.000000], [0.016000, 0.032000, 1.000000],
    [0.032000, 0.064000, 1.000000], [0.048000, 0.096000, 1.000000],
    [0.064000, 0.128000, 1.000000], [0.080000, 0.160000, 1.000000],
    [0.096000, 0.192000, 1.000000], [0.112000, 0.224000, 1.000000],
    [0.128000, 0.256000, 1.000000], [0.144000, 0.288000, 1.000000],
    [0.160000, 0.320000, 1.000000], [0.176000, 0.352000, 1.000000],
    [0.192000, 0.384000, 1.000000], [0.208000, 0.416000, 1.000000],
    [0.224000, 0.448000, 1.000000], [0.240000, 0.480000, 1.000000],
    [0.256000, 0.512000, 1.000000], [0.272000, 0.544000, 1.000000],
    [0.288000, 0.576000, 1.000000], [0.304000, 0.608000, 1.000000],
    [0.320000, 0.640000, 1.000000], [0.336000, 0.672000, 1.000000],
    [0.352000, 0.704000, 1.000000], [0.368000, 0.736000, 1.000000],
    [0.384000, 0.768000, 1.000000], [0.400000, 0.800000, 1.000000],
    [0.485714, 0.828571, 1.000000], [0.571429, 0.857143, 1.000000],
    [0.657143, 0.885714, 1.000000], [0.742857, 0.914286, 1.000000],
    [0.828571, 0.942857, 1.000000], [0.914286, 0.971429, 1.000000],
    [1.000000, 1.000000, 1.000000], [1.000000, 1.000000, 0.800000],
    [1.000000, 1.000000, 0.600000], [1.000000, 1.000000, 0.400000],
    [1.000000, 1.000000, 0.200000], [1.000000, 1.000000, 0.000000],
    [1.000000, 0.961538, 0.000000], [1.000000, 0.923077, 0.000000],
    [1.000000, 0.884615, 0.000000], [1.000000, 0.846154, 0.000000],
    [1.000000, 0.807692, 0.000000], [1.000000, 0.769231, 0.000000],
    [1.000000, 0.730769, 0.000000], [1.000000, 0.692308, 0.000000],
    [1.000000, 0.653846, 0.000000], [1.000000, 0.615385, 0.000000],
    [1.000000, 0.576923, 0.000000], [1.000000, 0.538462, 0.000000],
    [1.000000, 0.500000, 0.000000], [1.000000, 0.461538, 0.000000],
    [1.000000, 0.423077, 0.000000], [1.000000, 0.384615, 0.000000],
    [1.000000, 0.346154, 0.000000], [1.000000, 0.307692, 0.000000],
    [1.000000, 0.269231, 0.000000], [1.000000, 0.230769, 0.000000],
    [1.000000, 0.192308, 0.000000], [1.000000, 0.153846, 0.000000],
    [1.000000, 0.115385, 0.000000], [1.000000, 0.076923, 0.000000],
    [1.000000, 0.038462, 0.000000], [1.000000, 0.000000, 0.000000],
])

SURFACE_PREFIX_CANDIDATES: Sequence[str] = ("z", "h", "eta", "snap")
VELOCITY_PREFIX_PAIRS: Sequence[tuple[str, str]] = (("m", "n"), ("u", "v"))

def update_creation_time_windows(filepath: Path):
    """
    Manually set the file creation time on Windows to the current time.
    Requires ctypes and only works on Windows.
    """
    if os.name != 'nt' or ctypes is None:
        return

    try:
        # Get handle to file with write access to attributes
        FILE_WRITE_ATTRIBUTES = 0x0100
        OPEN_EXISTING = 3
        
        handle = ctypes.windll.kernel32.CreateFileW(
            str(filepath), FILE_WRITE_ATTRIBUTES, 0, None, OPEN_EXISTING, 0, None
        )
        
        if handle == -1: return

        # Convert current time to Windows FILETIME
        # epoch is Jan 1, 1601. 100-nanosecond intervals.
        timestamp = int((time.time() + 11644473600) * 10000000)
        filetime = wintypes.FILETIME(timestamp & 0xFFFFFFFF, timestamp >> 32)
        
        # Set CreationTime, AccessTime, and WriteTime.
        ctypes.windll.kernel32.SetFileTime(handle, ctypes.byref(filetime), None, None)
        ctypes.windll.kernel32.CloseHandle(handle)
    except Exception:
        pass

def contour_interval(mat: np.ndarray) -> int:
    vmin = float(np.nanmin(mat))
    if vmin > -500: return 10
    if vmin > -1000: return 50
    if vmin > -8000: return 100
    return 200

def load_grid(layer: str, dat_dir: Path) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    x_path = dat_dir / f"layer{layer}_x.dat"
    y_path = dat_dir / f"layer{layer}_y.dat"

    if not x_path.exists():
        print(f"CRITICAL ERROR: X-coordinate grid file not found at {x_path}")
        sys.exit(1)
    if not y_path.exists():
        print(f"CRITICAL ERROR: Y-coordinate grid file not found at {y_path}")
        sys.exit(1)

    x = np.loadtxt(x_path)
    y = np.loadtxt(y_path)
    nx, ny = len(x), len(y)
    
    bathy_path = dat_dir / f"layer{layer}.dat"
    if bathy_path.exists():
        bathy = np.loadtxt(bathy_path)
        bathy = np.reshape(bathy, (nx, ny), order="F")
    else:
        print(f"WARNING: Bathymetry file not found at {bathy_path}")
        print(" -> Shorelines (black borders) and depth contours will NOT be drawn.")
        bathy = None
        
    return x, y, bathy

def find_snapshot_indices(layer: str, dat_dir: Path, surface_prefix: Optional[str] = None) -> tuple[str, list[int]]:
    prefixes = [surface_prefix] if surface_prefix else list(SURFACE_PREFIX_CANDIDATES)
    for prefix in prefixes:
        if prefix is None: continue
        snaps: list[int] = []
        for path in dat_dir.glob(f"{prefix}_{layer}_*.dat"):
            tail = path.stem.split("_")[-1]
            if tail.isdigit():
                snaps.append(int(tail))
        if snaps:
            return prefix, sorted(snaps)
    return "", []

def load_flat(path: Path) -> np.ndarray:
    data = path.read_text()
    return np.fromstring(data, sep=" ", dtype=float)[:, None]

def load_snapshot(layer: str, snap: int, dat_dir: Path, surface_prefix: str, load_velocity: bool = False) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    suffix = f"{snap:06d}"
    surf_path = dat_dir / f"{surface_prefix}_{layer}_{suffix}.dat"
    
    if not surf_path.exists():
        print(f"WARNING: Surface elevation file not found: {surf_path}. Skipping this snapshot.")
        return None, None, None
        
    z = load_flat(surf_path)
    m_arr, n_arr = None, None

    if load_velocity:
        found_velocity = False
        for vx, vy in VELOCITY_PREFIX_PAIRS:
            vx_path = dat_dir / f"{vx}_{layer}_{suffix}.dat"
            vy_path = dat_dir / f"{vy}_{layer}_{suffix}.dat"
            if vx_path.exists() and vy_path.exists():
                m_arr = load_flat(vx_path)
                n_arr = load_flat(vy_path)
                found_velocity = True
                break
                
        if not found_velocity:
             print(f"WARNING: Velocity files requested but not found for snapshot {suffix}.")
        
    return z, m_arr, n_arr

def reshape_to_grid(arr: Optional[np.ndarray], nx: int, ny: int) -> Optional[np.ndarray]:
    if arr is None: return None
    return np.reshape(arr, (nx, ny), order="F")

def get_images_from_dir(layer: str, prefix: str, directory: Path) -> list[Path]:
    frames = sorted(directory.glob(f"{prefix}_{layer}_*.png"))
    if not frames:
        for p in SURFACE_PREFIX_CANDIDATES:
            frames = sorted(directory.glob(f"{p}_{layer}_*.png"))
            if frames: break
    if not frames:
        frames = sorted(directory.glob(f"snap_{layer}_*.jpg"))
    return frames

def write_mp4(frames: Sequence[Path], mp4_path: Path, fps: int) -> Path:
    if not frames:
        raise ValueError("No images found to build a video.")
    if imageio is None:
        raise RuntimeError("Install imageio and imageio-ffmpeg for MP4 support.")
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    
    with imageio.get_writer(mp4_path, fps=fps, macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))
    
    update_creation_time_windows(mp4_path)
    return mp4_path

def plot_snapshot(x, y, bathy, z, m, n, layer, snap, surface_prefix, cmap, stride, output) -> Path:
    nx, ny = len(x), len(y)
    z_grid = reshape_to_grid(z, nx, ny)
    m_grid = reshape_to_grid(m, nx, ny)
    n_grid = reshape_to_grid(n, nx, ny)

    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = ax.pcolormesh(x, y, z_grid.T, shading="auto", cmap=cmap, vmin=-12, vmax=12)
    fig.colorbar(mesh, ax=ax, label="Surface elevation")
    
    if m_grid is not None and n_grid is not None:
        step = slice(None, None, stride)
        ax.quiver(x[step], y[step], m_grid[step, step].T, n_grid[step, step].T, color="r", scale=1)

    if bathy is not None:
        dc = contour_interval(bathy)
        ax.contour(x, y, bathy.T, levels=np.arange(-8000, 0, dc), colors=[(0.5, 0.5, 0.5)], linewidths=0.6, linestyles='--')
        ax.contour(x, y, bathy.T, levels=[0], colors="k", linewidths=0.8)

    # Use FormatStrFormatter to ensure 2 decimal places and no scientific notation
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Use MaxNLocator to prevent tick overcrowding
    ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', prune=None))

    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Layer {layer} ({surface_prefix.upper()}) - Snapshot {snap:06d}")

    # Logic to detect overlap and rotate tick labels
    fig.canvas.draw() # Necessary to calculate positions
    
    # Check X-axis for overlap
    xticks = ax.get_xticklabels()
    if len(xticks) > 1:
        # Get bounding boxes in display coordinates
        bboxes = [t.get_window_extent() for t in xticks]
        overlap = any(bboxes[i].overlaps(bboxes[i+1]) for i in range(len(bboxes)-1))
        
        # Heuristic check: if the total width of labels exceeds 80% of axis width, rotate anyway
        total_label_width = sum(box.width for box in bboxes)
        axis_width = ax.get_window_extent().width
        
        if overlap or total_label_width > (axis_width * 0.8):
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            # Tighten layout again because rotation occupies more vertical space
            fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    
    update_creation_time_windows(output)
    return output

def main() -> None:
    parser = argparse.ArgumentParser(description="COMCOT Data Plotter & Video Generator")
    parser.add_argument("layer", help="Layer ID (e.g., 01)")
    parser.add_argument("snaptime", type=int, nargs="?", help="Specific snapshot index. Omit for latest.")
    parser.add_argument("--dat-dir", type=Path, default=Path("."), help="Data directory")
    parser.add_argument("--img-dir", type=Path, help="Image directory for video")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory")
    parser.add_argument("--surface-prefix", type=str, help="Prefix (z, h, etc.)")
    parser.add_argument("--arrows", action="store_true", help="Plot velocity arrows (Disabled by default)")
    parser.add_argument("--stride", type=int, default=10, help="Velocity arrow density")
    parser.add_argument("--all", action="store_true", help="Process all snapshots")
    parser.add_argument("--mp4", action="store_true", help="Generate MP4")
    parser.add_argument("--mp4-name", type=str, help="Custom MP4 name")
    parser.add_argument("--fps", type=int, default=6, help="Frame rate")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    parser.add_argument("--video-only", action="store_true", help="Video only mode")

    args = parser.parse_args()
    verbose = not args.quiet

    prefix, snaps = find_snapshot_indices(args.layer, args.dat_dir, args.surface_prefix)
    
    if args.video_only:
        search_dir = args.img_dir if args.img_dir else args.output_dir
        eff_prefix = prefix if prefix else (args.surface_prefix if args.surface_prefix else "z")
        frames = get_images_from_dir(args.layer, eff_prefix, search_dir)
        mp4_name = args.mp4_name if args.mp4_name else f"{eff_prefix}_{args.layer}.mp4"
        
        if verbose:
            print(f"Building video from {len(frames)} frames in: {search_dir}")
            
        video = write_mp4(frames, args.output_dir / mp4_name, args.fps)
        print(f"Video saved (timestamp updated): {video}")
        return

    if not snaps:
        print(f"CRITICAL ERROR: No .dat files found in {args.dat_dir}")
        sys.exit(1)

    target_snaps = snaps if args.all else ([args.snaptime] if args.snaptime else [snaps[-1]])
    x, y, bathy = load_grid(args.layer, args.dat_dir)
    cmap = ListedColormap(SEDIMENT_CMAP)

    if verbose:
        print(f"\n--- Starting Plot Process ---")
        print(f"Detected surface prefix: '{prefix}'")
        print(f"Reading .dat data from: {args.dat_dir.resolve()}")
        print(f"Saving plots to:        {args.output_dir.resolve()}")
        print(f"Processing {len(target_snaps)} snapshots...")

    saved_frames = []
    start_time = time.time()
    for i, snap in enumerate(target_snaps, 1):
        z, m, n = load_snapshot(args.layer, snap, args.dat_dir, prefix, load_velocity=args.arrows)
        if z is None: continue
        
        outfile = args.output_dir / f"{prefix}_{args.layer}_{snap:06d}.png"
        status_msg = " (creation time updated)" if outfile.exists() else ""
            
        saved = plot_snapshot(x, y, bathy, z, m, n, args.layer, snap, prefix, cmap, args.stride, outfile)
        saved_frames.append(saved)
        
        if verbose:
            print(f"[{i}/{len(target_snaps)}] Plot created: {outfile.name}{status_msg}")

    if verbose and saved_frames:
        elapsed = time.time() - start_time
        print(f"\nPlotting completed in {elapsed:.2f}s ({elapsed/len(saved_frames):.2f}s per plot)")

    if args.mp4:
        frames_to_use = saved_frames if saved_frames else get_images_from_dir(args.layer, prefix, args.output_dir)
        mp4_name = args.mp4_name if args.mp4_name else f"{prefix}_{args.layer}.mp4"
        mp4_path = args.output_dir / mp4_name
        
        video_status = " (creation time updated)" if mp4_path.exists() else ""
        
        if verbose:
            print(f"Stitching {len(frames_to_use)} frames into MP4...")
            
        video = write_mp4(frames_to_use, mp4_path, args.fps)
        print(f"Video saved: {video}{video_status}")

if __name__ == "__main__":
    main()
