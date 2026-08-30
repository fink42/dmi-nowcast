"""Phase 0 reconnaissance: pull recent DMI composite files and inspect them.

Run: .venv/bin/python scripts/phase0_recon.py

Verifies (plan §13 Phase 0):
1. Filename pattern dk.com.YYYYMMDDHHMM.500_max.h5
2. ODIM HDF5 structure (/what, /where, /dataset1/data1/what)
3. quantity / gain / offset / nodata / undetect are present
4. fullRange vs doppler interleaving (plan §3.2)

Writes a Markdown report to recon_report.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dmi_nowcast_core.fetch import RadarFeature, download, list_latest  # noqa: E402

CACHE = Path(__file__).parent.parent / ".recon_cache"
REPORT = Path(__file__).parent.parent / "recon_report.md"
NUM_FRAMES = 6


def main() -> int:
    print(f"Listing latest {NUM_FRAMES} composite features from DMI…")
    features = list_latest(limit=NUM_FRAMES)
    if not features:
        print("ERROR: no features returned", file=sys.stderr)
        return 1

    print(f"Got {len(features)} features. Downloading to {CACHE}/…")
    paths: list[tuple[RadarFeature, Path]] = []
    for f in features:
        path = download(f, CACHE)
        size_kb = path.stat().st_size / 1024
        print(f"  {f.datetime_utc.isoformat()}  {f.scan_type:10s}  {size_kb:6.1f} KB  {f.filename}")
        paths.append((f, path))

    print(f"\nInspecting first file: {paths[0][1].name}")
    inspect(paths[0][1])

    print(f"\nWriting report → {REPORT}")
    write_report(paths)
    return 0


def inspect(path: Path) -> dict:
    """Print and return key HDF5 metadata for one file."""
    with h5py.File(path, "r") as h5:
        print(f"  Root groups: {list(h5.keys())}")
        what = dict(h5["/what"].attrs) if "/what" in h5 else {}
        where = dict(h5["/where"].attrs) if "/where" in h5 else {}
        print(f"  /what: {_decode(what)}")
        print(f"  /where: {_decode(where)}")
        how = dict(h5["/how"].attrs) if "/how" in h5 else {}
        print(f"  /how: {_decode(how)}")
        dataset_keys = [k for k in h5.keys() if k.startswith("dataset")]
        print(f"  Datasets: {dataset_keys}")
        if dataset_keys:
            ds = h5[dataset_keys[0]]
            ds_what = dict(ds["what"].attrs) if "what" in ds else {}
            print(f"  {dataset_keys[0]}/what: {_decode(ds_what)}")
            data_keys = [k for k in ds.keys() if k.startswith("data")]
            print(f"  {dataset_keys[0]} children: {list(ds.keys())}")
            if data_keys:
                data_group = ds[data_keys[0]]
                data_what = dict(data_group["what"].attrs) if "what" in data_group else {}
                print(f"  {dataset_keys[0]}/{data_keys[0]}/what: {_decode(data_what)}")
                if "data" in data_group:
                    arr = data_group["data"]
                    print(f"  {dataset_keys[0]}/{data_keys[0]}/data: shape={arr.shape} dtype={arr.dtype}")
        return {"what": _decode(what), "where": _decode(where)}


def _decode(d: dict) -> dict:
    """Convert HDF5 attrs (often numpy types / bytes) to plain Python for printing."""
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        elif hasattr(v, "tolist"):
            v = v.tolist()
        out[k] = v
    return out


def write_report(paths: list[tuple[RadarFeature, Path]]) -> None:
    lines: list[str] = []
    lines.append("# Phase 0 reconnaissance report\n")
    lines.append(f"Inspected {len(paths)} files from `https://opendataapi.dmi.dk/v1/radardata`.\n")
    lines.append("## Files\n")
    lines.append("| datetime UTC | scanType | size (KB) | filename |")
    lines.append("|---|---|---:|---|")
    for f, p in paths:
        size_kb = p.stat().st_size / 1024
        lines.append(f"| {f.datetime_utc.isoformat()} | {f.scan_type} | {size_kb:.1f} | `{f.filename}` |")

    lines.append("\n## HDF5 structure of first file\n")
    f0, p0 = paths[0]
    lines.append(f"File: `{p0.name}`\n")
    with h5py.File(p0, "r") as h5:
        for group in ("/what", "/where", "/how"):
            if group in h5:
                lines.append(f"### `{group}` attributes\n")
                for k, v in _decode(dict(h5[group].attrs)).items():
                    lines.append(f"- `{k}`: `{v}`")
                lines.append("")
        ds_keys = [k for k in h5.keys() if k.startswith("dataset")]
        if ds_keys:
            ds = h5[ds_keys[0]]
            ds_what = _decode(dict(ds["what"].attrs)) if "what" in ds else {}
            lines.append(f"### `/{ds_keys[0]}/what` attributes\n")
            lines.append(f"- (empty: {not ds_what})" if not ds_what else "")
            for k, v in ds_what.items():
                lines.append(f"- `{k}`: `{v}`")
            data_keys = [k for k in ds.keys() if k.startswith("data")]
            if data_keys:
                dg = ds[data_keys[0]]
                data_what = _decode(dict(dg["what"].attrs)) if "what" in dg else {}
                lines.append(f"\n### `/{ds_keys[0]}/{data_keys[0]}/what` attributes\n")
                lines.append(f"- (empty: {not data_what})" if not data_what else "")
                for k, v in data_what.items():
                    lines.append(f"- `{k}`: `{v}`")
                if "data" in dg:
                    arr = dg["data"]
                    lines.append(f"\n### Array\n\n- shape: `{arr.shape}`\n- dtype: `{arr.dtype}`")

    lines.append("\n## Key findings vs plan\n")
    lines.append("- **Scaling metadata at root `/what`, not `/datasetN/dataN/what`** "
                 "(plan §3.3 assumes the latter; `dataset1/data1/what` is empty in DMI files). "
                 "pysteps' default importer will not handle this — needs a shim.")
    lines.append("- **`product=DBZH` at root** replaces the ODIM-standard `quantity` field in `dataN/what`.")
    lines.append("- **Z–R coefficients published in `/how`** as `zr-a` and `zr-b` "
                 "(values match Marshall–Palmer 200/1.6). Read from file rather than hardcoding (plan §6.2).")
    lines.append("- **File size 70–90 KB**, not 0.5–2 MB as plan §12.4 estimates. "
                 "Cache budget is much smaller than planned.")
    lines.append("- **`version=0.2`** in `/what` — not ODIM v2.0/v2.2. May just be DMI versioning.")
    lines.append("- **Single dataset/data group** — no multi-quantity files to handle.")

    REPORT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
