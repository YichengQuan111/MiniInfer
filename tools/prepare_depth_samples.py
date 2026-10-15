#!/usr/bin/env python3
"""Select a prediction-independent, versioned 16-image DA-2K smoke subset."""
import argparse
import hashlib
import io
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

ARCHIVE_SHA256 = "ff0e48e7cc53273efd1312610e51f1ec87bea0b8a22daf37125fd81246592b81"
REVISION = "528f83c26f02a6a7a11c5a036dfc9ee19907ebd9"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--offset-per-scene", type=int, default=0,
                        help="Skip this many eligible images per scene without reading predictions")
    parser.add_argument("--id-prefix", default="")
    args = parser.parse_args()
    if args.offset_per_scene < 0 or not all(c.isalnum() or c in "_-" for c in args.id_prefix):
        parser.error("offset must be nonnegative and id-prefix must contain only letters/digits/_/-")
    if args.manifest.exists():
        parser.error("manifest exists; select a new output path")
    with args.archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != ARCHIVE_SHA256:
        raise ValueError("DA-2K archive checksum mismatch")
    args.images.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    counts = defaultdict(int)
    eligible = defaultdict(int)
    skipped = []
    with zipfile.ZipFile(args.archive) as archive:
        annotations_bytes = archive.read("DA-2K/annotations.json")
        annotations = json.loads(annotations_bytes)
        for path, pairs in sorted(annotations.items()):
            scene = path.split("/")[1]
            if counts[scene] == 2:
                continue
            image_bytes = archive.read("DA-2K/" + path)
            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
                if max(width, height) > 4096 or width * height > 2097152:
                    skipped.append({"source": path, "reason": "exceeds frozen image limits"})
                    continue
                rgb = image.convert("RGB")
                eligible[scene] += 1
                if eligible[scene] <= args.offset_per_scene:
                    continue
                image_id = f"{args.id_prefix}{scene}_{counts[scene]}"
                target = args.images / f"{image_id}.ppm"
                encoded = f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes()
            if target.exists() and target.read_bytes() != encoded:
                raise ValueError(f"refusing to replace different sample: {target}")
            if not target.exists():
                target.write_bytes(encoded)
            translated = []
            for pair in pairs:
                if pair["closer_point"] not in ("point1", "point2"):
                    raise ValueError("unknown annotation direction")
                near = pair[pair["closer_point"]]
                far = pair["point2" if pair["closer_point"] == "point1" else "point1"]
                for y, x in (near, far):
                    if not (0 <= x < width and 0 <= y < height):
                        raise ValueError("annotation outside image")
                translated.append({"near": [near[1], near[0]], "far": [far[1], far[0]],
                                   "status": "valid", "evidence": "DA-2K independent closer_point label"})
            samples.append({"id": image_id,
                            "path": Path(os.path.relpath(target, args.manifest.parent)).as_posix(),
                            "split": "development" if counts[scene] == 0 else "final",
                            "sha256": hashlib.sha256(encoded).hexdigest(),
                            "source": f"DA-2K@{REVISION}/{path}",
                            "source_sha256": hashlib.sha256(image_bytes).hexdigest(),
                            "width": width, "height": height, "near_far_pairs": translated})
            counts[scene] += 1
    if len(counts) != 8 or any(value != 2 for value in counts.values()):
        raise ValueError(f"expected two eligible images from each of 8 scenes: {dict(counts)}")
    result = {"schema_version": 1, "annotation_version": 1,
              "source": "https://huggingface.co/datasets/depth-anything/DA-2K",
              "source_revision": REVISION, "archive_sha256": actual,
              "annotation_sha256": hashlib.sha256(annotations_bytes).hexdigest(),
              "selection": f"lexicographic 2 dimension-eligible images per scene after offset {args.offset_per_scene}; index 0 development, 1 final; no prediction-based filtering",
              "scope": "16-image smoke subset; not the official DA-2K benchmark score",
              "skipped": skipped, "samples": samples}
    args.manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(samples), "pairs": sum(len(s["near_far_pairs"]) for s in samples),
                      "scenes": dict(counts), "manifest": str(args.manifest)}))


if __name__ == "__main__":
    main()
