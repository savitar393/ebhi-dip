from pathlib import Path

project_root = Path(__file__).resolve().parent
dataset_root = project_root / "data" / "raw" / "EBHI-SEG"

total_images = 0
total_masks = 0
total_pairs = 0

categories = sorted(
    folder for folder in dataset_root.iterdir()
    if folder.is_dir()
)

for category in categories:
    images = {file.name for file in (category / "image").glob("*.png")}
    masks = {file.name for file in (category / "label").glob("*.png")}

    pairs = images & masks

    print(
        f"{category.name}: "
        f"images={len(images)}, masks={len(masks)}, pairs={len(pairs)}"
    )

    for name in sorted(images - masks):
        print(f"  Missing mask: {name}")

    for name in sorted(masks - images):
        print(f"  Missing image: {name}")

    total_images += len(images)
    total_masks += len(masks)
    total_pairs += len(pairs)

print(
    f"\nTOTAL: images={total_images}, "
    f"masks={total_masks}, pairs={total_pairs}"
)