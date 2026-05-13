"""
Generate synthetic demo data for cross-modal retrieval development.
Creates dummy images with matching captions for immediate pipeline testing.
No external dataset download required.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# Demo captions organized by scene category
SCENES = {
    "outdoor": [
        "a dog running in a green park",
        "a red car parked on the street",
        "people walking on a sandy beach",
        "mountains covered in snow under blue sky",
        "a cat sitting on a garden wall",
        "children playing football in the park",
        "a bird flying over the ocean at sunset",
        "trees with golden leaves in autumn",
        "a bicycle leaning against a wooden fence",
        "flowers blooming in a sunny meadow",
    ],
    "indoor": [
        "a person reading a book on a sofa",
        "a laptop on a wooden desk near a window",
        "a cup of coffee on a kitchen table",
        "a cat sleeping on a cozy armchair",
        "bookshelves filled with colorful books",
        "a painting hanging on a white wall",
        "a person cooking in a bright kitchen",
        "a guitar resting against a bed frame",
        "a desk lamp illuminating a workspace",
        "an orange cat looking out the window",
    ],
    "nature": [
        "a deer standing in a misty forest",
        "waves crashing on rocky cliffs",
        "a rainbow appearing after heavy rain",
        "a field of sunflowers under bright sun",
        "a river flowing through green valleys",
        "a butterfly resting on a purple flower",
        "snowflakes falling on pine trees",
        "a full moon rising over a calm lake",
        "lightning striking over dark mountains",
        "coral reef with colorful fish underwater",
    ],
}

ID_COUNTER = [0]


def create_demo_image(image_id: int, scene: str, caption: str, size: int = 224):
    """Create a synthetic image with colored patterns and caption text."""
    category_colors = {
        "outdoor": (135, 206, 235),   # sky blue
        "indoor": (255, 228, 196),     # bisque
        "nature": (144, 238, 144),     # light green
    }
    base_color = category_colors.get(scene, (200, 200, 200))

    img = Image.new("RGB", (size, size), base_color)

    # Add some random shapes to make each image unique
    draw = ImageDraw.Draw(img)
    np.random.seed(image_id)

    for _ in range(5):
        x1 = np.random.randint(10, size - 20)
        y1 = np.random.randint(10, size - 20)
        x2 = x1 + np.random.randint(20, 80)
        y2 = y1 + np.random.randint(20, 80)
        color = tuple(np.random.randint(50, 200, 3))
        if np.random.random() > 0.5:
            draw.ellipse([x1, y1, x2, y2], fill=color)
        else:
            draw.rectangle([x1, y1, x2, y2], fill=color)

    return img


def generate_demo_dataset(output_dir: str = "data/demo",
                          num_images_per_scene: int = 15):
    """Generate a complete synthetic cross-modal dataset."""
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for scene, captions in SCENES.items():
        selected_captions = captions[:num_images_per_scene]

        for i, caption in enumerate(selected_captions):
            image_id = ID_COUNTER[0]
            ID_COUNTER[0] += 1

            # Create and save image
            img = create_demo_image(image_id, scene, caption)
            img_path = img_dir / f"{image_id:05d}.jpg"
            img.save(img_path)

            records.append({
                "image_id": image_id,
                "image_path": str(img_path),
                "caption": caption,
                "scene": scene,
                "year": np.random.choice([2021, 2022, 2023, 2024]),
                "source": np.random.choice(["flickr", "unsplash", "pexels"]),
            })

    # Save as CSV and JSON
    df = pd.DataFrame(records)
    csv_path = output_dir / "annotations.csv"
    df.to_csv(csv_path, index=False)

    json_path = output_dir / "annotations.json"
    records_for_json = []
    for _, r in df.iterrows():
        records_for_json.append({
            "image_id": int(r["image_id"]),
            "image_path": str(r["image_path"]),
            "caption": r["caption"],
            "scene": r["scene"],
            "year": int(r["year"]),
            "source": r["source"],
        })
    with open(json_path, "w") as f:
        json.dump(records_for_json, f, indent=2)

    print(f"[demo] Generated {len(records)} image-text pairs in {output_dir}/")
    print(f"  Images: {img_dir}/ ({len(list(img_dir.glob('*.jpg')))} files)")
    print(f"  Annotations: {csv_path}")
    print(f"  Scenes: {df['scene'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    generate_demo_dataset(num_images_per_scene=15)
