import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import glob
import sys
import datetime

# Try to import imageio for video generation
try:
    import imageio.v2 as imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("Warning: 'imageio' library not found. MP4 generation will be disabled.")
    print("To enable, run: pip install imageio imageio-ffmpeg")

# --- 1. DEFINISI COLORMAP (SEDIMENT) ---
# Hardcoded sediment colormap (Deep Blue -> White -> Red)
SEDIMENT_DATA = np.array([
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
    [1.000000, 1.000000, 1.000000], # WHITE CENTER
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

SEDIMENT_CMAP = ListedColormap(SEDIMENT_DATA, name='sediment')

def read_binary_snapshot(fn):
    """Reads COMCOT binary snapshot data and associated coordinates."""
    path, fn0 = os.path.split(fn)
    if not path: path = '.'
        
    try:
        underscore_index = fn0.find('_')
        if underscore_index != -1:
            ilayer_str = fn0[underscore_index+1 : underscore_index+3]
            if ilayer_str.isdigit():
                ilayer = int(ilayer_str)
            else:
                ilayer = 1
        else:
            ilayer = 1
    except Exception:
        ilayer = 1

    layer_str = f"{ilayer:02d}"
    x_file = os.path.join(path, f'_xcoordinate{layer_str}.dat')
    y_file = os.path.join(path, f'_ycoordinate{layer_str}.dat')

    try:
        x = np.loadtxt(x_file)
        y = np.loadtxt(y_file)
    except OSError:
        return None, None, None

    # Grid Staggering
    if fn0.lower().startswith(('m', 'n')):
        if fn0.lower().startswith('m'):
            x = x[:-1] + 0.5 * (x[1] - x[0])
        elif fn0.lower().startswith('n'):
            y = y[:-1] + 0.5 * (y[1] - y[0])

    try:
        with open(fn, 'rb') as fid:
            def read_fortran_record(dtype, count=1):
                rec_len_start = np.fromfile(fid, dtype=np.int32, count=1)
                if rec_len_start.size == 0: return None
                data = np.fromfile(fid, dtype=dtype, count=count)
                rec_len_end = np.fromfile(fid, dtype=np.int32, count=1)
                return data

            fp_arr = read_fortran_record(np.int32, 1)
            fp = fp_arr[0]
            real_type = np.float32 if fp == 4 else np.float64

            dims = read_fortran_record(np.int32, 2)
            ncol, nrow = dims[0], dims[1]

            dat = np.zeros((nrow, ncol), dtype=real_type)
            for i in range(nrow):
                row_data = read_fortran_record(real_type, ncol)
                dat[i, :] = row_data

    except Exception as e:
        print(f"Error reading binary file {fn}: {e}")
        return None, None, None

    return x, y, dat

def format_title_string(filename):
    """Converts filename to 'Layer-[Type] [Num], hh:mm:ss'"""
    base = os.path.basename(filename)
    name_no_ext = os.path.splitext(base)[0]
    parts = name_no_ext.split('_')
    
    l_type = parts[0] if len(parts) > 0 else '?'
    l_num = parts[1] if len(parts) > 1 else '01'
    time_val_str = parts[2] if len(parts) > 2 else '0'
    
    try:
        seconds = int(time_val_str)
        time_formatted = str(datetime.timedelta(seconds=seconds))
        if len(time_formatted.split(':')) == 2: 
            time_formatted = "0:" + time_formatted
    except ValueError:
        time_formatted = time_val_str

    return f"Layer-{l_type.upper()} {l_num}, {time_formatted}"

def get_masked_data(fn, bathymetry_data):
    x, y, b = read_binary_snapshot(fn)
    if b is None: return None, None, None

    xb, yb, h = bathymetry_data
    
    fname = os.path.basename(fn)
    is_flux = fname.lower().startswith(('m', 'n'))

    if not is_flux and h is not None and b.shape == h.shape:
        mask = (h <= 0) & ((b + h) <= 0)
        b[mask] = np.nan
            
    return x, y, b

def contour_interval(min_val):
    if min_val > -500: return 10
    if min_val > -1000: return 50
    if min_val > -8000: return 100
    return 200

def save_snapshot_to_png(fn, output_folder, bathymetry_data, sym_limit):
    x, y, b = get_masked_data(fn, bathymetry_data)
    if b is None: return None

    xb, yb, h = bathymetry_data

    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    extent = [x[0]-dx/2, x[-1]+dx/2, y[0]-dy/2, y[-1]+dy/2]

    # Plot Data (Symmetric vmin/vmax for white zero)
    im = ax.imshow(b, extent=extent, origin='lower', cmap=SEDIMENT_CMAP, 
                   aspect='auto', vmin=-sym_limit, vmax=sym_limit)
    cbar = plt.colorbar(im)
    cbar.set_label('Surface Elevation (m)')

    # Plot Bathymetry Contours
    if h is not None and x.shape == h.shape:
        min_depth = np.nanmin(h)
        dc = contour_interval(min_depth)
        
        # Negative Contours (Depth)
        levels = np.arange(-8000, 0, dc)
        levels = levels[levels >= min_depth]
        if len(levels) > 0:
            ax.contour(x, y, h, levels=levels, colors=[(0.5, 0.5, 0.5)], 
                       linewidths=0.6, linestyles='--')

        # Coastline (Zero)
        ax.contour(x, y, h, levels=[0], colors='k', linewidths=1.0, zorder=5)

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'k-', linewidth=1.5)

    title_str = format_title_string(fn)
    ax.set_title(title_str, fontsize=14)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.tick_params(axis='x', rotation=45) 
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    out_name = os.path.splitext(os.path.basename(fn))[0] + ".png"
    out_path = os.path.join(output_folder, out_name)
    
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path

def create_video_from_images(image_files, output_path, fps=6):
    """Creates an MP4 video from a list of image paths."""
    if not IMAGEIO_AVAILABLE:
        print("Cannot create video: imageio not installed.")
        return

    if not image_files:
        print("No images found to create video.")
        return

    print(f"\nStitching {len(image_files)} images into video at {fps} FPS...")
    try:
        with imageio.get_writer(output_path, fps=fps) as writer:
            for img_path in image_files:
                image = imageio.imread(img_path)
                writer.append_data(image)
        print(f"Video saved successfully: {output_path}")
    except Exception as e:
        print(f"Error creating video: {e}")

def main():
    print("--- COMCOT Plotter & Animator (White-Zero Scale) ---")
    
    # 1. Inputs
    input_folder = input("Enter Input Folder Path (default: ./): ").strip() or "./"
    output_folder = input("Enter Output Folder Path (default: ./output): ").strip() or "./output"
    file_prefix = input("Enter File Prefix (default: z): ").strip() or "z"
    layer_num = input("Enter Layer Number (default: 01): ").strip() or "01"
    
    # Video specific inputs
    make_video = False
    video_fps = 6
    if IMAGEIO_AVAILABLE:
        vid_input = input("Create MP4 video after plotting? (y/n, default: y): ").strip().lower() or 'y'
        if vid_input == 'y':
            make_video = True
            fps_input = input("Enter Video FPS (default: 6): ").strip() or "6"
            video_fps = int(fps_input)

    layer_num = f"{int(layer_num):02d}"
    pattern = os.path.join(input_folder, f"{file_prefix}_{layer_num}_*.dat")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching pattern: {pattern}")
        return

    # 2. Load Bathymetry
    print(f"\nLoading bathymetry for Layer {layer_num}...")
    bath_file = os.path.join(input_folder, f'_bathymetry{layer_num}.dat')
    if os.path.exists(bath_file):
        xb, yb, h = read_binary_snapshot(bath_file)
        bathymetry_data = (xb, yb, h)
    else:
        print("Warning: Bathymetry file not found. Land masking disabled.")
        bathymetry_data = (None, None, None)

    # 3. Scan for Max Amplitude (Symmetric Scale)
    print("Scanning files to determine Absolute Max Amplitude...")
    max_abs_val = 0.0
    for i, f in enumerate(files):
        print(f"Scanning [{i+1}/{len(files)}] ...", end='\r')
        _, _, b = get_masked_data(f, bathymetry_data)
        if b is not None:
            current_max_abs = np.nanmax(np.abs(b))
            if current_max_abs > max_abs_val:
                max_abs_val = current_max_abs
    
    if max_abs_val == 0: max_abs_val = 0.1 
    print(f"\nGlobal Symmetric Limit determined: +/- {max_abs_val:.4f}")

    # 4. Export Images
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    generated_images = []
    print("\nStarting Image Export...")
    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        print(f"[{i+1}/{len(files)}] Exporting {filename} ...", end='\r')
        
        # Save png and keep track of the path
        saved_path = save_snapshot_to_png(file_path, output_folder, bathymetry_data, max_abs_val)
        if saved_path:
            generated_images.append(saved_path)
    
    print(f"\nImages saved to: {output_folder}")

    # 5. Create Video
    if make_video and generated_images:
        video_name = f"{file_prefix}_{layer_num}_animation.mp4"
        video_path = os.path.join(output_folder, video_name)
        create_video_from_images(generated_images, video_path, fps=video_fps)

    print("\nDone!")

if __name__ == "__main__":
    main()
