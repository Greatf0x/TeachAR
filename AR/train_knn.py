import os
import glob
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Expecting gesture samples saved as .npz inside data/gestures/
# Each .npz should contain:
#   X: (N, D) feature vectors
#   y: (N,) labels (strings)
GESTURE_DIR = "data/gestures"
MODEL_OUT = "data/models/knn.pkl"

def load_dataset():
    files = sorted(glob.glob(os.path.join(GESTURE_DIR, "*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No .npz gesture files found in {GESTURE_DIR}. "
            "Run record_gestures.py first to collect data."
        )

    Xs, ys = [], []
    for f in files:
        data = np.load(f, allow_pickle=True)
        if "X" not in data or "y" not in data:
            raise ValueError(f"{f} must contain arrays 'X' and 'y'.")

        X = data["X"]
        y = data["y"]
        Xs.append(X)
        ys.append(y)

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y

def main():
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    X, y = load_dataset()
    print("Loaded dataset:", X.shape, y.shape)
    print("Classes:", sorted(set(y.tolist())))

    # Split for quick sanity evaluation
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # KNN: tune n_neighbors based on your data size
    knn = KNeighborsClassifier(n_neighbors=3, weights="distance")
    knn.fit(Xtr, ytr)

    pred = knn.predict(Xte)
    print("\nConfusion Matrix:\n", confusion_matrix(yte, pred))
    print("\nReport:\n", classification_report(yte, pred))

    joblib.dump(knn, MODEL_OUT)
    print(f"\nSaved model to: {MODEL_OUT}")

if __name__ == "__main__":
    main()