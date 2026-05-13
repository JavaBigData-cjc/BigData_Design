"""
Download real Flickr30k dataset for cross-modal retrieval experiments.
Sources: HuggingFace datasets (nlphuji/flickr30k) or direct S3 mirror.

Flickr30k: 31,783 images, 5 captions each = 158,915 text-image pairs.
Size: ~4.4 GB zip, ~5 GB extracted.
"""

import sys
import zipfile
import subprocess
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_via_huggingface(output_dir: str = "data/raw/flickr30k"):
    """
    Download Flickr30k via HuggingFace datasets library.
    This fetches image URLs and captions. Images must be downloaded separately.
    Best for getting metadata/captions quickly.
    """
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/2] Loading flickr30k dataset from HuggingFace...")
    ds = load_dataset("nlphuji/flickr30k", split="train")

    print(f"  Loaded {len(ds)} samples")
    print(f"  Columns: {ds.column_names}")
    print(f"  Sample: {ds[0]}")

    # Save captions as CSV
    import pandas as pd
    records = []
    for i, item in enumerate(ds):
        for j, caption in enumerate(item["caption"]):
            records.append({
                "image_id": i,
                "caption_index": j,
                "caption": caption,
                "image_path": item.get("image", ""),
                "image_url": item.get("image_url", ""),
            })

    df = pd.DataFrame(records)
    csv_path = output_dir / "captions.csv"
    df.to_csv(csv_path, index=False)

    unique_images = df[["image_id", "image_url"]].drop_duplicates()
    print(f"  [OK] {len(unique_images)} unique images, {len(df)} captions")
    print(f"  Captions saved to {csv_path}")

    return df


def download_via_s3(output_dir: str = "data/raw/flickr30k"):
    """
    Download the full Flickr30k dataset (images + captions) from AutoGluon S3 mirror.
    URL: https://automl-mm-bench.s3.amazonaws.com/flickr30k.zip
    Size: ~4.38 GB
    """
    import os
    import requests
    from tqdm import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = "https://automl-mm-bench.s3.amazonaws.com/flickr30k.zip"
    zip_path = output_dir / "flickr30k.zip"

    # Check if already downloaded
    if zip_path.exists() and zip_path.stat().st_size > 100_000_000:
        print(f"  [skip] {zip_path.name} already exists ({zip_path.stat().st_size / 1e9:.1f} GB)")
    else:
        print(f"[1/3] Downloading flickr30k.zip from AutoGluon S3...")
        print(f"  URL: {url}")
        print(f"  Destination: {zip_path}")
        print(f"  Size: ~4.4 GB (may take 10-30 minutes depending on network)")

        # Use requests with streaming
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with open(zip_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True,
                      desc="  Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"  [OK] Downloaded {zip_path.stat().st_size / 1e9:.1f} GB")

    # Extract
    extract_dir = output_dir / "images"
    if extract_dir.exists() and len(list(extract_dir.glob("*.jpg"))) > 1000:
        print(f"  [skip] Already extracted: {len(list(extract_dir.glob('*.jpg')))} images")
    else:
        print(f"[2/3] Extracting {zip_path.name}...")
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract only image files
            image_files = [f for f in zf.namelist()
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            for f in tqdm(image_files, desc="  Extracting"):
                zf.extract(f, extract_dir)

        print(f"  [OK] Extracted {len(image_files)} images to {extract_dir}/")

    # Parse captions
    caption_file = extract_dir / "results.csv"
    if not caption_file.exists():
        # Try alternative location
        for f in extract_dir.glob("**/*.csv"):
            caption_file = f
            break

    if caption_file.exists():
        print(f"[3/3] Loading captions from {caption_file}")
        import pandas as pd
        df = pd.read_csv(caption_file, delimiter="|")
        df.to_csv(output_dir / "captions.csv", index=False)
        print(f"  [OK] {len(df)} captions loaded")
    else:
        print("[3/3] No captions CSV found in archive, using HuggingFace captions")
        download_via_huggingface(output_dir)

    return output_dir


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Download Flickr30k dataset")
    ap.add_argument("--source", choices=["huggingface", "s3", "both"],
                    default="s3", help="Download source (default: s3)")
    ap.add_argument("--output", default="data/raw/flickr30k",
                    help="Output directory")
    args = ap.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "huggingface":
        download_via_huggingface(str(output_dir))
    elif args.source in ("s3", "both"):
        download_via_s3(str(output_dir))
        if args.source == "both":
            download_via_huggingface(str(output_dir))

    print(f"\nDone! Data saved to {output_dir}/")
    print("Next step: python scripts/run_data_pipeline.py --source flickr30k")


if __name__ == "__main__":
    main()
