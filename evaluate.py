from ultralytics import YOLO
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\dronr\Desktop\YOLO11-Object-Detection-with-COCO128-main"
)

DATASET = Path(
    r"C:\Users\dronr\Desktop\YOLO11-Object-Detection-with-COCO128-main\datasets\coco128.yaml"
)

MODEL_1 = Path(
    r"C:\Users\dronr\Desktop\YOLO11-Object-Detection-with-COCO128-main\runs\detect\train\weights\best.pt"
)

MODEL_2 = Path(
    r"C:\Users\dronr\Desktop\YOLO11-Object-Detection-with-COCO128-main\runs\detect\train-2\weights\best.pt"
)


# ============================================================
# CHECK FILES
# ============================================================

print("Checking files...")
print()

print("Dataset :", DATASET)
print("Model 1 :", MODEL_1)
print("Model 2 :", MODEL_2)

print()


if not DATASET.exists():
    raise FileNotFoundError(
        f"Dataset YAML not found:\n{DATASET}"
    )


if not MODEL_1.exists():
    raise FileNotFoundError(
        f"Model 1 not found:\n{MODEL_1}"
    )


if not MODEL_2.exists():
    raise FileNotFoundError(
        f"Model 2 not found:\n{MODEL_2}"
    )


print("All required files found successfully!")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_path):

    print("\n" + "=" * 70)
    print(f"Evaluating: {model_path}")
    print("=" * 70)

    # Load model
    model = YOLO(str(model_path))

    # Evaluate using OUR dataset YAML
    metrics = model.val(
        data=str(DATASET),
        split="val",
        verbose=False
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision = float(metrics.box.mp)

    recall = float(metrics.box.mr)

    map50 = float(metrics.box.map50)

    map50_95 = float(metrics.box.map)

    # --------------------------------------------------------
    # F1 Score
    # --------------------------------------------------------

    if precision + recall > 0:

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "mAP50": map50,
        "mAP50-95": map50_95
    }


# ============================================================
# EVALUATE MODEL 1
# ============================================================

results_1 = evaluate_model(MODEL_1)


# ============================================================
# EVALUATE MODEL 2
# ============================================================

results_2 = evaluate_model(MODEL_2)


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n")

print("=" * 80)
print("                         MODEL COMPARISON")
print("=" * 80)

print(
    f"{'Metric':<15}"
    f"{'train/best.pt':<25}"
    f"{'train-2/best.pt':<25}"
)

print("-" * 80)


metrics_to_compare = [
    "Precision",
    "Recall",
    "F1 Score",
    "mAP50",
    "mAP50-95"
]


for metric in metrics_to_compare:

    value1 = results_1[metric]

    value2 = results_2[metric]

    print(
        f"{metric:<15}"
        f"{value1:<25.4f}"
        f"{value2:<25.4f}"
    )


print("=" * 80)


# ============================================================
# BEST MODEL BY METRIC
# ============================================================

print("\nBEST MODEL BY METRIC")

print("=" * 80)


for metric in metrics_to_compare:

    value1 = results_1[metric]

    value2 = results_2[metric]


    if value1 > value2:

        winner = "train/best.pt"

    elif value2 > value1:

        winner = "train-2/best.pt"

    else:

        winner = "Both models"


    print(
        f"{metric:<15}: {winner}"
    )


print("=" * 80)