import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

def read_binary_snapshot(fn):
    """
    Reads COMCOT binary snapshot data and associated coordinates.
    """
    path, fn0 = os.path.split(fn)
    if not path: path = '.'
        
    # Parse Layer ID
    try:
        # Standard case: 'z_01_000000.dat' -> find first '_', take next 2 chars
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

    # Load Coordinates
    x_file = os.path.join(path, f'_xcoordinate{layer_str}.dat')
    y_file = os.path.join(path, f'_ycoordinate{layer_str}.dat')

    try:
        x = np.loadtxt(x_file)
        y = np.loadtxt(y_file)
    except OSError:
        # Fallback if specific layer coord not found, try layer 1 or warn
        print(f"Warning: Could not find coordinates for layer {layer_str}, skipping {fn0}")
        return None, None, None

    # Handle Grid Staggering
    if fn0.lower().startswith(('m', 'n')):
        if fn0.lower().startswith('m'):
            x = x[:-1] + 0.5 * (x[1] - x[0])
        elif fn0.lower().startswith('n'):
            y = y[:-1] + 0.5 * (y[1] - y[0])

    # Read Binary Data
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

def save_snapshot_to_png(fn, output_folder):
    """
    Plots the snapshot and saves it to the output folder.
    """
    # 1. Read Data
    x, y, b = read_binary_snapshot(fn)
    if b is None: return

    # 2. Read Bathymetry (for masking and coastline)
    path, dataFile = os.path.split(fn)
    if not path: path = '.'
    
    # Extract layer from filename (e.g., z_01_000000.dat -> 01)
    parts = dataFile.split('_')
    if len(parts) >= 2:
        ilayer = parts[1]
    else:
        ilayer = '01'

    bath_file = os.path.join(path, f'_bathymetry{ilayer}.dat')
    
    # Try to read bathymetry; if missing, proceed without it
    xb, yb, h = read_binary_snapshot(bath_file)
    
    # 3. Apply Masking
    is_flux = dataFile.lower().startswith(('m', 'n'))
    if not is_flux and h is not None:
        mask = (h <= 0) & ((b + h) <= 0)
        b[mask] = np.nan

    # 4. Setup Plot
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Determine Extents
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    extent = [x[0]-dx/2, x[-1]+dx/2, y[0]-dy/2, y[-1]+dy/2]

    # Plot Data
    # Use generic min/max or fixed range if preferred
    im = ax.imshow(b, extent=extent, origin='lower', cmap='jet', aspect='auto')
    plt.colorbar(im)

    # Plot Coastline (if bathymetry exists)
    if h is not None:
        ax.contour(x, y, h, levels=[0], colors='k', linewidths=1.0)

    # Plot Bounding Box
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'k-', linewidth=1.5)

    ax.set_title(dataFile)
    ax.set_aspect('equal')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    # 5. Save and Close
    out_name = os.path.splitext(dataFile)[0] + ".png"
    out_path = os.path.join(output_folder, out_name)
    
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig) # Important: free memory

def main():
    print("--- COMCOT Data to PNG Exporter ---")
    
    # 1. Take Inputs
    input_folder = input("Enter Input Folder Path (default: ./): ").strip() or "./"
    output_folder = input("Enter Output Folder Path (default: ./output): ").strip() or "./output"
    file_prefix = input("Enter File Prefix (default: z): ").strip() or "z"
    layer_num = input("Enter Layer Number (default: 01): ").strip() or "01"

    # Normalize layer number format (ensure 2 digits)
    layer_num = f"{int(layer_num):02d}"

    # 2. List Files
    # Pattern: input_folder/prefix_layer_*.dat
    pattern = os.path.join(input_folder, f"{file_prefix}_{layer_num}_*.dat")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching pattern: {pattern}")
        return

    # 3. Define Steps
    # Analyze the numeric part of the filenames to determine the step size
    indices = []
    for f in files:
        base = os.path.basename(f)
        # Assuming format: z_01_000060.dat
        # Split by '_' -> ['z', '01', '000060.dat']
        parts = base.split('_')
        try:
            # Extract number before .dat
            idx_part = parts[-1].replace('.dat', '')
            indices.append(int(idx_part))
        except ValueError:
            continue

    if len(indices) >= 2:
        step_diff = indices[1] - indices[0]
        print(f"\nFound {len(files)} files.")
        print(f"File Index Start: {indices[0]}")
        print(f"File Index End:   {indices[-1]}")
        print(f"Detected Step:    {step_diff}")
    else:
        print(f"\nFound {len(files)} file(s).")

    # Confirm
    confirm = input("Start export? (y/n): ").lower()
    if confirm != 'y':
        print("Aborted.")
        return

    # Create output directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 4. Loop Export Process
    print("\nStarting Export...")
    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        print(f"[{i+1}/{len(files)}] Processing {filename} ...", end='\r')
        
        save_snapshot_to_png(file_path, output_folder)
    
    print(f"\n\nDone! Images saved to: {output_folder}")

if __name__ == "__main__":
    main()
