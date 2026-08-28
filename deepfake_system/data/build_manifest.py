"""Scan the extracted datasets and build one unified manifest.

The four archives all use different folder layouts, so this script infers
label / video-id / method from the path and writes a single CSV:

    source,video_id,identity,label,method,frame_path

Run it with --dry-run first. It prints the layout it detected so you can
fix the patterns below before committing to a 30-minute scan.

    python -m data.build_manifest --dry-run
    python -m data.build_manifest
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Path tokens that mark a clip as authentic. Checked case-insensitively.
REAL_TOKENS = (
    "original", "originals", "real", "youtube", "actors", "pristine",
    "celeb-real", "celeb_real", "youtube-real", "youtube_real", "genuine",
)

# Tokens that mark a clip as manipulated, grouped by generator family so
# the evaluation can report per-family behaviour.
FAKE_METHODS = (
    "deepfakes", "face2face", "faceswap", "neuraltextures", "faceshifter",
    "celeb-synthesis", "celeb_synthesis", "simswap", "inswapper", "blendface",
    "uniface", "e4s", "facedancer", "fsgan", "mobileswap", "danet",
    "wav2lip", "sadtalker", "mraa", "fomm", "tpsm", "styleheat",
    "sd15", "sdxl", "ddim", "ddpm", "stargan", "stylegan", "collab",
    "vqgan", "pixart", "midjourney", "heygen", "hyperreenact",
    # Singular. Some FF++ redistributions name the folder "deepfake", which
    # the plural token above does not match. Kept last so archives using
    # "deepfakes" still report under that name.
    "deepfake",
)


def _norm(p: Path) -> str:
    return str(p).lower().replace("\\", "/")


def infer_label(path: Path) -> tuple[int, str]:
    """Return (label, method). label 1 = fake, 0 = real."""
    s = _norm(path)
    for m in FAKE_METHODS:
        if m in s:
            return 1, m
    for t in REAL_TOKENS:
        if t in s:
            return 0, "real"
    if re.search(r"/(fake|manipulated|synth|generated)(/|_)", s):
        return 1, "unknown_fake"
    if re.search(r"/(real|authentic)(/|_)", s):
        return 0, "real"
    return -1, "unresolved"


def infer_video_id(path: Path, source: str) -> str:
    """Frames of one video normally sit in one folder; fall back to the
    numeric stem prefix when they are flat files."""
    parent = path.parent.name
    if re.fullmatch(r"[A-Za-z0-9_\-]{2,}", parent) and parent.lower() not in {
        "frames", "faces", "images", "crops"
    }:
        return f"{source}/{path.parent.parent.name}/{parent}"
    stem = re.sub(r"[-_]?(frame)?[-_]?\d{1,6}$", "", path.stem)
    return f"{source}/{parent}/{stem or path.stem}"


# Splits are often already baked into the archive layout. FF++ unpacks as
# dataset_split/test/deepfake/008_990/... and Celeb-DF as test/fake/<hash>/...
# Both name the split as a whole path component.
_SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}


def infer_split(path: Path) -> str | None:
    """Return the split named in the path, or None if the layout has none."""
    for part in _norm(path).split("/")[:-1]:
        if part in _SPLIT_ALIASES:
            return _SPLIT_ALIASES[part]
    return None


def infer_identity(video_id: str) -> str:
    """FF++ uses 000_003 (target_source); Celeb-DF uses idNN_xxxx.
    Identity grouping stops the same face landing in train and test."""
    tail = video_id.rsplit("/", 1)[-1]
    m = re.match(r"(id\d+)", tail)
    if m:
        return m.group(1)
    m = re.match(r"(\d{3})[_-]\d{3}", tail)
    if m:
        return m.group(1)
    return tail


def scan(root: Path, source: str, limit: int | None = None):
    rows, unresolved = [], []
    files = (p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    for i, p in enumerate(files):
        if limit and i >= limit:
            break
        label, method = infer_label(p)
        if label < 0:
            unresolved.append(p)
            continue
        vid = infer_video_id(p, source)
        rows.append({
            "source": source,
            "video_id": vid,
            "identity": infer_identity(vid),
            "label": label,
            "method": method,
            "frame_path": str(p),
            "path_split": infer_split(p),
        })
    return rows, unresolved


def assign_splits(rows, use_path_splits=True):
    """Split on identity so no face appears in two splits.

    When an archive already ships a train/val/test layout AND every one of
    its files sits under one of those folders, that split is used verbatim
    for that source. Two reasons to prefer it:

      * It is the split the dataset was published with, so numbers stay
        comparable to other work.
      * Identity grouping only works when the folder name carries the
        identity. FF++ does (008_990 -> 008). Celeb-DF redistributions that
        hash the folder name do not, so every video looks like its own
        identity and the "identity split" silently degrades into a video
        split. Deferring to the shipped layout is the honest option.

    Sources without a usable layout fall back to identity assignment.
    """
    rng = random.Random(DATA.seed)
    by_source = defaultdict(set)
    for r in rows:
        by_source[r["source"]].add((r["identity"], r["label"]))

    # Which sources carry a complete split layout?
    path_split_ok = {}
    for source in by_source:
        rows_s = [r for r in rows if r["source"] == source]
        path_split_ok[source] = (use_path_splits and
                                 all(r.get("path_split") for r in rows_s))

    split_of = {}
    for source, ids in by_source.items():
        if source in DATA.holdout_sources:
            for ident, _ in ids:
                split_of[(source, ident)] = "holdout"
            continue
        if path_split_ok[source]:
            continue                      # handled per-row below
        ids = sorted(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_val = max(1, int(n * DATA.val_ratio))
        n_test = max(1, int(n * DATA.test_ratio))
        for j, (ident, _) in enumerate(ids):
            if j < n_val:
                s = "val"
            elif j < n_val + n_test:
                s = "test"
            else:
                s = "train"
            split_of[(source, ident)] = s

    for r in rows:
        if (r["source"] not in DATA.holdout_sources
                and path_split_ok[r["source"]]):
            s = r["path_split"]
        else:
            s = split_of[(r["source"], r["identity"])]
        # Unseen-generator probe: pull these methods out of train entirely.
        if s == "train" and r["method"] in DATA.holdout_methods:
            s = "unseen_method"
        r["split"] = s
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="scan 4000 files per source and print what was found")
    ap.add_argument("--root", default=str(DATA.root))
    ap.add_argument("--out", default=str(DATA.manifest))
    ap.add_argument("--ignore-path-splits", action="store_true",
                    help="ignore any train/val/test folders in the archive "
                         "and assign splits by identity instead")
    args = ap.parse_args()

    root = Path(args.root)
    all_rows, all_unresolved = [], []
    for source, folder in DATA.sources.items():
        d = root / folder
        if not d.exists():
            print(f"[skip] {d} not found")
            continue
        rows, unres = scan(d, source, limit=4000 if args.dry_run else None)
        print(f"[{source}] {len(rows):,} frames, {len(unres):,} unresolved")
        all_rows += rows
        all_unresolved += unres[:20]

    if not all_rows:
        print("Nothing found. Check --root and DATA.sources in config.py.")
        return

    print("\nlabel counts :", Counter(r["label"] for r in all_rows))
    print("methods      :", Counter(r["method"] for r in all_rows).most_common(15))
    print("videos       :", len({r["video_id"] for r in all_rows}))
    if all_unresolved:
        print("\nCould not label these paths — add a token to REAL_TOKENS or "
              "FAKE_METHODS:")
        for p in all_unresolved[:10]:
            print("   ", p)

    if args.dry_run:
        print("\nDry run only. Re-run without --dry-run to write the manifest.")
        return

    all_rows = assign_splits(all_rows, use_path_splits=not args.ignore_path_splits)
    print("splits       :", Counter(r["split"] for r in all_rows))
    for source in sorted({r["source"] for r in all_rows}):
        rows_s = [r for r in all_rows if r["source"] == source]
        mode = ("archive layout" if all(r.get("path_split") for r in rows_s)
                and not args.ignore_path_splits else "identity grouping")
        n_ident = len({r["identity"] for r in rows_s})
        print(f"  {source:10s} split by {mode:16s} "
              f"({n_ident:,} identities over {len({r['video_id'] for r in rows_s}):,} videos)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "source", "video_id", "identity", "label", "method",
            "split", "frame_path"], extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {len(all_rows):,} rows -> {out}")


if __name__ == "__main__":
    main()
