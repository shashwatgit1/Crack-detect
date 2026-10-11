from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Locate dataset directory flexibly (handling 'Dataset' vs 'dataset')
DATASET_DIR = PROJECT_ROOT / "Dataset" / "SDNET2018"
if not DATASET_DIR.exists():
    DATASET_DIR = PROJECT_ROOT / "dataset" / "SDNET2018"

SPLIT_FILE = PROJECT_ROOT / "dataset" / "splits.csv"
if not SPLIT_FILE.parent.exists():
    SPLIT_FILE = PROJECT_ROOT / "Dataset" / "splits.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_label(path):
    """
    Infer the binary crack label from folder name.
    Returns 1 for Cracked, 0 for Non-cracked (Uncracked), or None.
    """
    for part in reversed(path.parts):
        name = part.lower().replace("_", "").replace("-", "").replace(" ", "")

        # SDNET2018 conventions: CD, CP, CW = Cracked (Decks, Pavements, Walls)
        if name in {"cd", "cp", "cw"}:
            return 1

        # SDNET2018 conventions: UD, UP, UW = Uncracked (Decks, Pavements, Walls)
        if name in {"ud", "up", "uw"}:
            return 0

        # General descriptive conventions
        if name in {"cracked", "crack", "positive"}:
            return 1

        if name in {"noncracked", "noncrack", "uncracked", "negative"}:
            return 0

    return None


def get_category_and_surface(path):
    """
    Extract category (D, P, W) and surface group ID from path.
    SDNET2018 filenames follow '<surface_id>-<patch_id>.jpg'.
    Grouping by surface prevents data leakage between splits.
    """
    category = "unknown"
    for part in path.parts:
        if part.upper() in {"D", "P", "W"}:
            category = part.upper()
            break

    stem = path.stem
    surface_prefix = stem.split("-")[0] if "-" in stem else stem
    surface_id = f"{category}_{surface_prefix}"
    return category, surface_id


def discover_images():
    """Find labeled images under the SDNET2018 directory."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}\n"
            "Place the extracted SDNET2018 dataset inside dataset/SDNET2018."
        )

    records = []
    for path in DATASET_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            label = get_label(path)
            if label is not None:
                category, surface_id = get_category_and_surface(path)
                records.append({
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "label": label,
                    "category": category,
                    "surface_id": surface_id
                })

    if not records:
        raise RuntimeError(
            f"No labeled images found in {DATASET_DIR}. Expected subfolders such as "
            "D/CD, D/UD, P/CP, P/UP, W/CW, W/UW."
        )

    df = pd.DataFrame(records).drop_duplicates(subset=["path"])

    if df["label"].nunique() != 2:
        raise RuntimeError(
            "Both cracked and non-cracked images are required in the dataset."
        )

    return df


def create_splits(force=False):
    """
    Create a surface-grouped 70/15/15 train/validation/test split.
    Group-splitting by surface ensures patches from the same concrete structure
    do not leak across train, validation, and test sets.
    """
    if SPLIT_FILE.exists() and not force:
        return pd.read_csv(SPLIT_FILE)

    df = discover_images()
    df["split"] = ""

    # Group-based split per category to maintain category representation
    rng = np.random.RandomState(42)

    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        surfaces = np.array(sorted(cat_df["surface_id"].unique()))
        rng.shuffle(surfaces)

        n_surfaces = len(surfaces)
        n_train = int(round(0.70 * n_surfaces))
        n_val = int(round(0.15 * n_surfaces))

        train_surfs = set(surfaces[:n_train])
        val_surfs = set(surfaces[n_train:n_train + n_val])
        test_surfs = set(surfaces[n_train + n_val:])

        df.loc[(df["category"] == cat) & (df["surface_id"].isin(train_surfs)), "split"] = "train"
        df.loc[(df["category"] == cat) & (df["surface_id"].isin(val_surfs)), "split"] = "validation"
        df.loc[(df["category"] == cat) & (df["surface_id"].isin(test_surfs)), "split"] = "test"

    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPLIT_FILE, index=False)

    print(f"Saved split file to: {SPLIT_FILE}")
    print("\nImages per split and class:")
    print(pd.crosstab(df["split"], df["label"], margins=True))

    return df


class CrackDataset(Dataset):
    """PyTorch dataset for concrete crack classification."""

    def __init__(self, split="train", transform=None):
        self.transform = transform
        self.df = create_splits()

        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split '{split}'. Expected 'train', 'validation', or 'test'.")

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if self.df.empty:
            raise RuntimeError(f"No images found for split: {split}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = PROJECT_ROOT / Path(row["path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["label"])
        return image, label


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset dir:  {DATASET_DIR}")
    print("\nGenerating leakage-free splits...")
    create_splits(force=True)
