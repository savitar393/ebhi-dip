import csv
from pathlib import Path

import numpy as np

SEED = 42
CANDIDATES = 10_000
SPLITS = ("train", "val", "test")

project_root = Path(__file__).resolve().parent
metadata_dir = project_root / "data" / "metadata"

with (metadata_dir / "image_hashes.csv").open(
    newline="", encoding="utf-8-sig"
) as file:
    records = [
        row for row in csv.DictReader(file)
        if row["pair_status"] == "paired"
    ]

# Give capitalization variants the same provisional group ID.
for row in records:
    row["group_id"] = (
        Path(row["image_path"]).stem.split("-")[0].upper()
    )

categories = sorted({row["category"] for row in records})
group_names = sorted({row["group_id"] for row in records})

category_index = {
    name: index for index, name in enumerate(categories)
}
group_index = {
    name: index for index, name in enumerate(group_names)
}

# Each row describes one group's image counts across categories.
group_counts = np.zeros(
    (len(group_names), len(categories)),
    dtype=np.int64,
)

for row in records:
    group_counts[
        group_index[row["group_id"]],
        category_index[row["category"]],
    ] += 1

category_totals = group_counts.sum(axis=0)
targets = np.array([0.70, 0.15, 0.15])

n_train_groups = round(0.70 * len(group_names))
n_val_groups = round(0.15 * len(group_names))

rng = np.random.default_rng(SEED)
best_score = float("inf")
best_parts = None
best_counts = None

for _ in range(CANDIDATES):
    order = rng.permutation(len(group_names))

    parts = np.split(
        order,
        [n_train_groups, n_train_groups + n_val_groups],
    )

    counts = np.stack([
        group_counts[indices].sum(axis=0)
        for indices in parts
    ])

    # Require every category in train, validation, and test.
    if np.any(counts == 0):
        continue

    fractions = counts / category_totals
    score = np.mean((fractions - targets[:, None]) ** 2)

    if score < best_score:
        best_score = float(score)
        best_parts = [indices.copy() for indices in parts]
        best_counts = counts.copy()

if best_parts is None:
    raise SystemExit("No candidate contained every category in every subset.")

group_assignment = {
    group_names[index]: split_name
    for split_name, indices in zip(SPLITS, best_parts)
    for index in indices
}

for row in records:
    row["split"] = group_assignment[row["group_id"]]

# Check group separation using the resulting sample assignments.
group_sets = [
    {
        row["group_id"]
        for row in records
        if row["split"] == split_name
    }
    for split_name in SPLITS
]

overlap = sum(
    len(group_sets[i] & group_sets[j])
    for i in range(3)
    for j in range(i + 1, 3)
)

if overlap:
    raise SystemExit("Group overlap detected; candidate was not saved.")

output_path = metadata_dir / "split_candidate.csv"

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

print(f"{'Category':<20} {'Train':>7} {'Val':>7} {'Test':>7}")

for index, category in enumerate(categories):
    train, val, test = best_counts[:, index]
    print(f"{category:<20} {train:>7} {val:>7} {test:>7}")

print("\nSubset totals:")
for split_name, counts, groups in zip(SPLITS, best_counts, group_sets):
    total = int(counts.sum())
    print(
        f"  {split_name}: images={total} "
        f"({100 * total / len(records):.2f}%), "
        f"groups={len(groups)}"
    )

print("\nGroups shared between subsets:", overlap)
print("Saved:", output_path)