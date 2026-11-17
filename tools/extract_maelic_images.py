"""
Extract the original HICO-DET JPEGs from the maelic/HICO-DET-coco-format
parquet shards into the layout expected by datasets/hico.py.

Each row has `file_name` (e.g. HICO_train2015_00000001.jpg) and `image`
(a HuggingFace image struct with `bytes`/`path`). The bytes are written
unchanged, so the filenames stay aligned with our converted annotations.
"""

import argparse
import glob
import json
import os
import sys

import pyarrow.parquet as pq


def iter_rows(path, batch_size=64):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["file_name", "image"]):
        names = batch.column("file_name").to_pylist()
        images = batch.column("image").to_pylist()
        for name, image in zip(names, images):
            yield name, image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dir", default="data/maelic_parquet")
    parser.add_argument("--out_root", default="data/hico_20160224_det/images")
    parser.add_argument("--train_anno", default="data/hico_20160224_det/annotations/trainval_hico.json")
    parser.add_argument("--test_anno", default="data/hico_20160224_det/annotations/test_hico.json")
    parser.add_argument("--summary", default="data/maelic_parquet/extract_summary.json")
    args = parser.parse_args()

    train_anno = json.load(open(args.train_anno))
    test_anno = json.load(open(args.test_anno))
    train_needed = {a["file_name"] for a in train_anno if a["hoi_annotation"]}
    test_needed = {a["file_name"] for a in test_anno}

    summary = {}
    for split, target_name, needed in [
        ("train", "train2015", train_needed),
        ("test", "test2015", test_needed),
    ]:
        target_dir = os.path.join(args.out_root, target_name)
        os.makedirs(target_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(args.parquet_dir, split + "-*.parquet")))
        if not files:
            raise FileNotFoundError("no %s parquet shards in %s" % (split, args.parquet_dir))

        written = set()
        total = 0
        for shard in files:
            shard_count = 0
            for name, image in iter_rows(shard):
                if not name:
                    continue
                data = image.get("bytes") if isinstance(image, dict) else None
                if not data:
                    continue
                with open(os.path.join(target_dir, name), "wb") as f:
                    f.write(data)
                written.add(name)
                total += 1
                shard_count += 1
                if total % 2000 == 0:
                    print("%s: %d images..." % (split, total), flush=True)
            print("%s: %s -> %d images" % (split, os.path.basename(shard), shard_count), flush=True)

        missing = sorted(needed - written)
        extra = sorted(written - needed)
        summary[split] = {
            "written": len(written),
            "needed": len(needed),
            "missing": missing,
            "extra": extra,
            "total_rows_written": total,
        }
        print("%s: written %d, needed %d, missing %d, extra %d" % (
            split, len(written), len(needed), len(missing), len(extra)), flush=True)
        if missing:
            miss_path = os.path.join(args.parquet_dir, "missing_%s.json" % split)
            json.dump(missing, open(miss_path, "w"))
            print("%s: missing list -> %s" % (split, miss_path), flush=True)

    json.dump(summary, open(args.summary, "w"), indent=2)
    print("summary -> %s" % args.summary, flush=True)


if __name__ == "__main__":
    main()
