import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE

def train_hybrid_models(
    embeddings_path: str = "embeddings.npy",
    dataset_path: str = "dataset_semantic.csv",
    label_encoder_path: str = "label_encoder.pkl"
):
    print("=" * 75)
    print("      HYBRID SEMANTIC ML TRAINING PIPELINE (SCIKIT-LEARN)       ")
    print("=" * 75)

    # 1. Load Data Artifacts
    print(f"[*] Loading embeddings from: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    print(f"[*] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"[*] Loading label encoder from: {label_encoder_path}")
    le = joblib.load(label_encoder_path)

    # 2. Build Hybrid Feature Matrix (Dense 384-D Embeddings + 4 Structural Features)
    structural_cols = ["text_length", "num_options", "avg_time_taken", "past_accuracy"]
    for col in structural_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    structural_features = df[structural_cols].values
    X_hybrid = np.hstack([embeddings, structural_features])
    
    # Target Labels
    if "label" in df.columns:
        y = df["label"].values
    else:
        y = le.transform(df["difficulty"])

    print(f"[+] Hybrid Feature Matrix X_hybrid Shape: {X_hybrid.shape} (384 dense vector + 4 structural)")
    print(f"[+] Target Classes: {le.classes_}")

    # Feature Names
    embedding_feature_names = [f"emb_{i}" for i in range(embeddings.shape[1])]
    all_feature_names = embedding_feature_names + structural_cols
    joblib.dump(all_feature_names, "hybrid_feature_names.pkl")
    print(f"[+] Saved feature names ({len(all_feature_names)}) to hybrid_feature_names.pkl")

    # 3. Train/Test Split (Stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_hybrid, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4. Define 5 Scikit-Learn Classifiers
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM-RBF": SVC(kernel="rbf", probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "MLP(128,64)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    }

    # 5. 5-Fold Stratified Cross Validation & Evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_dict = {}
    test_scores_dict = {}
    trained_models = {}

    print("\n" + "-" * 75)
    print(f"{'Model Name':<22} | {'5-Fold CV Accuracy':<18} | {'Test Accuracy':<14}")
    print("-" * 75)

    best_model_name = None
    best_test_acc = -1.0

    for name, model in models.items():
        # Cross validation on full dataset
        cv_scores = cross_val_score(model, X_hybrid, y, cv=cv, scoring="accuracy")
        mean_cv = cv_scores.mean()
        cv_scores_dict[name] = mean_cv

        # Train on train set & evaluate on test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_scores_dict[name] = test_acc
        trained_models[name] = model

        print(f"{name:<22} | {mean_cv*100:>16.2f}% | {test_acc*100:>12.2f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_name = name

    print("-" * 75)
    print(f"\n[BEST MODEL] {best_model_name} with Test Accuracy = {best_test_acc*100:.2f}%")

    best_model = trained_models[best_model_name]
    # Retrain best model on full dataset for maximum generalization
    best_model.fit(X_hybrid, y)
    joblib.dump(best_model, "hybrid_semantic_model.pkl")
    print(f"[+] Saved best model ({best_model_name}) to hybrid_semantic_model.pkl")

    # 6. Full Classification Report for Best Model
    y_test_pred = trained_models[best_model_name].predict(X_test)
    print("\n" + "=" * 70)
    print(f"CLASSIFICATION REPORT FOR BEST MODEL ({best_model_name}):")
    print("=" * 70)
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))
    print("=" * 70)

    # 7. Generate & Save Visualizations
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Model Comparison (Test vs CV Accuracy)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_pos = np.arange(len(models))
    width = 0.35

    ax.bar(x_pos - width/2, [cv_scores_dict[m]*100 for m in models], width, label="5-Fold CV Acc (%)", color="#1E88E5")
    ax.bar(x_pos + width/2, [test_scores_dict[m]*100 for m in models], width, label="Test Set Acc (%)", color="#00E5FF")

    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Hybrid Semantic ML: 5-Model Benchmark Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(models.keys()), fontsize=10, fontweight="bold")
    ax.legend(frameon=True)
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig("hybrid_model_comparison.png", dpi=300)
    plt.close()
    print("[+] Saved hybrid_model_comparison.png")

    # Plot 2: Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted Difficulty Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Difficulty Label", fontsize=11, fontweight="bold")
    ax.set_title(f"Confusion Matrix: Best Hybrid Model ({best_model_name})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("hybrid_confusion.png", dpi=300)
    plt.close()
    print("[+] Saved hybrid_confusion.png")

    # Plot 3: t-SNE of Semantic Embeddings
    print("[*] Computing 2D t-SNE projection of 384-D semantic embeddings...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter_without_progress=300)
    embeddings_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    palette = {"Easy": "#2E7D32", "Medium": "#F57C00", "Hard": "#C62828"}
    difficulty_labels = le.inverse_transform(y)

    for diff in ["Easy", "Medium", "Hard"]:
        mask = (difficulty_labels == diff)
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=palette.get(diff, "#1E88E5"),
            label=diff,
            alpha=0.75,
            edgecolors="none",
            s=40
        )

    ax.set_title("t-SNE Latent Space: 384-D Semantic Embeddings Clustered by Difficulty", fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax.legend(title="Difficulty", frameon=True)
    plt.tight_layout()
    plt.savefig("semantic_tsne.png", dpi=300)
    plt.close()
    print("[+] Saved semantic_tsne.png")

if __name__ == "__main__":
    train_hybrid_models()
