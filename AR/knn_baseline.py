# knn_baseline.py
import glob, os, numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

DATA_DIR = "data/raw"
MODEL_DIR = "data/models"
os.makedirs(MODEL_DIR, exist_ok=True)

T = 32  # fixed temporal length for features

def load_dataset():
    X, y = [], []
    for label_dir in glob.glob(os.path.join(DATA_DIR, "*")):
        if not os.path.isdir(label_dir):
            continue
        label = os.path.basename(label_dir)
        for f in glob.glob(os.path.join(label_dir, "*.npy")):
            seq = np.load(f)  # shape (t,21,2)
            # pad/crop to T
            if seq.shape[0] >= T:
                seq2 = seq[:T]
            else:
                pad = np.repeat(seq[-1][None,...], T - seq.shape[0], axis=0)
                seq2 = np.concatenate([seq, pad], axis=0)
            X.append(seq2.reshape(-1))
            y.append(label)
    return np.array(X), np.array(y)

def main():
    X, y = load_dataset()
    if len(y) < 2:
        print("Need at least 2 labeled clips across labels in data/raw")
        return
    knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
    knn.fit(X, y)
    joblib.dump(knn, os.path.join(MODEL_DIR, "knn.pkl"))
    print(f"Trained KNN on {len(y)} samples across {len(set(y))} labels.")

if __name__ == "__main__":
    main()
