from ultralytics import YOLO
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(
    r"C:\Users\dronr\Desktop\YOLO11-Object-Detection-with-COCO128-main\yolo11n.pt"
)

DATASET_PATH = Path(
    r"C:\Users\dronr\Desktop\YOLO11-Object-Detection-with-COCO128-main\datasets\coco128.yaml"
)


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO11n model not found:\n{MODEL_PATH}"
    )

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset YAML not found:\n{DATASET_PATH}"
    )


print("=" * 70)
print("YOLO11n MODEL EVALUATION")
print("=" * 70)

print("Model  :", MODEL_PATH)
print("Dataset:", DATASET_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(str(MODEL_PATH))


# ============================================================
# VALIDATE MODEL
# ============================================================

metrics = model.val(
    data=str(DATASET_PATH),
    split="val",
    verbose=False
)


# ============================================================
# EXTRACT METRICS
# ============================================================

precision = float(metrics.box.mp)

recall = float(metrics.box.mr)

map50 = float(metrics.box.map50)

map50_95 = float(metrics.box.map)


# ============================================================
# F1 SCORE
# ============================================================

if precision + recall > 0:

    f1_score = (
        2 * precision * recall
        / (precision + recall)
    )

else:

    f1_score = 0.0


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("YOLO11n EVALUATION RESULTS")
print("=" * 70)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1_score:.4f}")
print(f"mAP50     : {map50:.4f}")
print(f"mAP50-95  : {map50_95:.4f}")

print("=" * 70)