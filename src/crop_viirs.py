"""
Crop the global VIIRS annual nighttime-lights composite down to Kenya's
bounding box, without ever decompressing (or holding) the full ~11.6GB
global raster.

The source file is a gzip-compressed BigTIFF with its IFD stored at the very
end (a "data first, metadata last" streaming layout) -- confirmed by
inspection: 16-byte header followed immediately by a single contiguous,
uncompressed float32 raster, row-major, 86400 x 33600 pixels covering
180W-180E, 75N-65S at 15 arc-second (1/240 deg) resolution.

Because the layout is fully predictable from geometry alone, we don't need
the IFD at all: we stream-decompress sequentially, skip (discard, never
store) everything before Kenya's row range, keep only Kenya's rows/columns,
and stop reading as soon as we're past them -- so we never decompress the
back half of the file, let alone write the full raster to disk.
"""
import gzip
import numpy as np

SRC_PATH = "data/raw/viirs/VNL_npp_2025_global_vcmslcfg_v2_c202604011200.average_masked.dat.tif.gz"
OUT_PATH = "data/raw/viirs/kenya_viirs_ntl.npz"

GLOBAL_WIDTH = 86400
GLOBAL_HEIGHT = 33600
PIXELS_PER_DEG = 240  # 15 arc-second
TOP_LAT = 75.0
LEFT_LON = -180.0
HEADER_BYTES = 16
ROW_BYTES = GLOBAL_WIDTH * 4  # float32

# Kenya bounding box with a safety margin
LON_MIN, LON_MAX = 33.5, 42.0
LAT_MIN, LAT_MAX = -5.5, 5.5

CHUNK = 16 * 1024 * 1024  # 16MB skip-read chunks


def lonlat_to_rowcol():
    col_start = int(round((LON_MIN - LEFT_LON) * PIXELS_PER_DEG))
    col_end = int(round((LON_MAX - LEFT_LON) * PIXELS_PER_DEG))
    row_start = int(round((TOP_LAT - LAT_MAX) * PIXELS_PER_DEG))  # higher lat = smaller row
    row_end = int(round((TOP_LAT - LAT_MIN) * PIXELS_PER_DEG))
    return row_start, row_end, col_start, col_end


def skip_bytes(f, n):
    remaining = n
    while remaining > 0:
        chunk = f.read(min(CHUNK, remaining))
        if not chunk:
            raise EOFError("Stream ended while skipping")
        remaining -= len(chunk)


def main():
    row_start, row_end, col_start, col_end = lonlat_to_rowcol()
    n_rows = row_end - row_start
    n_cols = col_end - col_start
    print(f"Kenya crop: rows {row_start}-{row_end} ({n_rows}), cols {col_start}-{col_end} ({n_cols})")
    print(f"Output size: ~{n_rows*n_cols*4/1e6:.1f} MB")

    out = np.empty((n_rows, n_cols), dtype=np.float32)

    with gzip.open(SRC_PATH, "rb") as f:
        f.read(HEADER_BYTES)

        skip_to_first_row = row_start * ROW_BYTES
        print(f"Skipping {skip_to_first_row/1e9:.2f} GB of rows before Kenya...")
        skip_bytes(f, skip_to_first_row)

        print("Reading Kenya's rows...")
        for i in range(n_rows):
            row_bytes = f.read(ROW_BYTES)
            if len(row_bytes) != ROW_BYTES:
                raise EOFError(f"Short read on row {row_start+i}")
            row = np.frombuffer(row_bytes, dtype="<f4")
            out[i] = row[col_start:col_end]
            if i % 500 == 0:
                print(f"  row {i}/{n_rows}")
        # Don't bother reading the rest of the file (rows after Kenya, IFD) --
        # we have everything we need.

    lons = LEFT_LON + (col_start + np.arange(n_cols) + 0.5) / PIXELS_PER_DEG
    lats = TOP_LAT - (row_start + np.arange(n_rows) + 0.5) / PIXELS_PER_DEG

    np.savez_compressed(OUT_PATH, radiance=out, lons=lons, lats=lats)
    print(f"Saved cropped Kenya VIIRS raster to {OUT_PATH}")
    print(f"radiance stats: min={out.min():.3f} max={out.max():.3f} mean={out.mean():.3f}")
    print(f"nonzero fraction: {(out>0).mean():.4f}")


if __name__ == "__main__":
    main()
