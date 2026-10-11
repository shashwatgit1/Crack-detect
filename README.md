# Concrete Crack Detection and Severity Estimation for Campus Infrastructure

An end-to-end computer vision application designed to detect cracks in concrete infrastructure (bridge decks, pavements, and walls), segment visible crack pixels, and estimate image-based crack severity.

---

## 1. Project Architecture

The system operates across a 4-stage pipeline:

```
[Concrete Image] 
       │
       ▼
[Stage A: Preprocessing & Classification] (MobileNetV3-Large)
       │  ↳ Output: "Cracked" vs "Non-cracked", confidence score
       ▼
[Stage B: Crack Segmentation] (U-Net)
       │  ↳ Output: Binary crack mask (256×256)
       ▼
[Stage C: Severity Estimation] (Morphological Skeletonization & Connected Components)
       │  ↳ Output: Crack area %, skeleton length, component count, severity tier (Minor / Moderate / High)
       ▼
[Stage D: Interactive Streamlit UI]
       ↳ Visual inspection, crack overlay, and metrics dashboard
```

---

## 2. Dataset Setup

### SDNET2018 (Classification)
Place the extracted SDNET2018 dataset inside `Dataset/SDNET2018/`. The directory structure adheres to standard benchmark conventions:
- **`D/` (Bridge Decks):** `CD` (Cracked Decks) and `UD` (Uncracked Decks)
- **`P/` (Pavements):** `CP` (Cracked Pavements) and `UP` (Uncracked Pavements)
- **`W/` (Walls):** `CW` (Cracked Walls) and `UW` (Uncracked Walls)

**Data Leakage Prevention:** Patches originating from the same surface photograph share a source ID prefix (e.g. `7001-115.jpg` and `7001-1.jpg`). `src/dataset.py` groups patches by surface so related patches never span across training, validation, and test splits.

### Campus Dataset (Segmentation)
- Photographs: `Dataset/campus/images/`
- Ground-truth binary masks: `Dataset/campus/annotations/` (White = crack, Black = background)

Use `python -m src.prepare_campus` to verify pairs or auto-generate initial candidate masks.

---

## 3. Installation & Setup

Ensure Python 3.10+ is installed and a virtual environment is activated:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

---

## 4. Execution Workflow

### Step 1: Generate Leakage-Free Dataset Splits
```powershell
python -m src.dataset
```
Creates `dataset/splits.csv` (70% train, 15% validation, 15% test) grouped by concrete surface.

### Step 2: Train the Classification Model
```powershell
python -m src.train_classifier --epochs 5 --batch-size 32
```
Trains MobileNetV3-Large with transfer learning and inverse-frequency class-weighted cross-entropy loss. Saves the best checkpoint to `models/classifier.pth`.

### Step 3: Evaluate Classifier
```powershell
python -m src.evaluate_classifier
```
Evaluates on the held-out test split, prints balanced accuracy and macro F1, and saves the confusion matrix to `results/classification/confusion_matrix.png`.

### Step 4: Prepare Campus Segmentation Annotations
```powershell
# Verify campus pairs:
python -m src.prepare_campus --verify

# Auto-generate candidate masks for newly added campus photos:
python -m src.prepare_campus --auto-annotate
```

### Step 5: Train the U-Net Segmentation Model
```powershell
python -m src.train_segmentation
```
Trains U-Net using combined BCE and Dice loss. Saves the best checkpoint to `models/segmentation.pth`.

### Step 6: Evaluate Segmentation
```powershell
python -m src.evaluate_segmentation
```
Computes Dice coefficient and IoU on validation data and saves visual overlays to `results/segmentation/qualitative_evaluation.png`.

### Step 7: Launch the Streamlit Web Application
```powershell
python -m streamlit run app/app.py
```
Open `http://localhost:8501` to upload concrete surface images and view real-time classification, crack segmentation overlays, and severity estimation.

---

## 5. Important Engineering Limitations

- **Visual vs. Structural Severity:** Image-based severity reflects 2D visual geometry (pixel area, skeleton length, component count) and **is not a structural safety diagnosis**.
- **Calibration Required:** Converting pixel dimensions to physical millimeters requires camera distance, perspective, and optical calibration targets.
- **On-site Inspection:** Engineering conclusions require certified civil and structural inspection.
