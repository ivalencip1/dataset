import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


# -----------------------------
# Utilities / plotting
# -----------------------------
RANDOM_STATE_DEFAULT = 42


def set_matplotlib_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 120


def get_iris_dataframe() -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    y = iris.target.copy()
    feature_names = list(iris.feature_names)
    target_names = list(iris.target_names)
    return X, y, feature_names, target_names


def make_model(
    model_name: str,
    *,
    scale: bool,
    lr_C: float,
    lr_max_iter: int,
    svm_C: float,
    svm_kernel: str,
    svm_gamma: str,
    svm_degree: int,
    knn_k: int,
    dt_max_depth: Optional[int],
    rf_n_estimators: int,
    rf_max_depth: Optional[int],
) -> Pipeline:
    """
    Returns a sklearn Pipeline (optionally with StandardScaler) to keep things consistent.
    """
    if model_name == "Logistic Regression":
        clf = LogisticRegression(
            C=lr_C,
            max_iter=lr_max_iter,
            multi_class="auto",
            solver="lbfgs",
        )
    elif model_name == "SVM (SVC)":
        # probability=True enables predict_proba (needed for ROC/PR/probability surfaces)
        clf = SVC(
            C=svm_C,
            kernel=svm_kernel,
            gamma=svm_gamma,
            degree=svm_degree,
            probability=True,
        )
    elif model_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=knn_k)
    elif model_name == "Decision Tree":
        clf = DecisionTreeClassifier(max_depth=dt_max_depth, random_state=RANDOM_STATE_DEFAULT)
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=RANDOM_STATE_DEFAULT,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", clf))
    return Pipeline(steps)


def compute_metrics(y_true, y_pred, average: str = "macro") -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def plot_confusion_matrix(y_true, y_pred, class_names: List[str], normalize: bool):
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    fig, ax = plt.subplots(figsize=(6, 4.6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión" + (" (Normalizada)" if normalize else ""))
    fig.tight_layout()
    return fig


def _safe_predict_proba(model: Pipeline, X: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns predict_proba output if available, otherwise None.
    """
    try:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        # Pipeline case: check last step
        if hasattr(model[-1], "predict_proba"):
            return model.predict_proba(X)
    except Exception:
        return None
    return None


def _safe_decision_function(model: Pipeline, X: np.ndarray) -> Optional[np.ndarray]:
    try:
        if hasattr(model, "decision_function"):
            return model.decision_function(X)
        if hasattr(model[-1], "decision_function"):
            return model.decision_function(X)
    except Exception:
        return None
    return None


def get_scores_for_curves(model: Pipeline, X: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    For ROC/PR in multiclass, we want per-class scores.
    Prefer predict_proba; else use decision_function.
    Returns (scores, kind) where kind in {"proba","decision"}.
    """
    proba = _safe_predict_proba(model, X)
    if proba is not None:
        return proba, "proba"
    dec = _safe_decision_function(model, X)
    if dec is not None:
        # decision_function for multiclass can be shape (n_samples, n_classes)
        # if binary may be (n_samples,), but iris is 3-class; still guard:
        if dec.ndim == 1:
            dec = np.vstack([-dec, dec]).T
        return dec, "decision"
    raise RuntimeError("El modelo no expone predict_proba ni decision_function; no se pueden dibujar curvas ROC/PR.")


def plot_roc_ovr(y_true: np.ndarray, scores: np.ndarray, class_names: List[str]):
    """
    One-vs-rest ROC curves. scores shape (n_samples, n_classes)
    """
    y_bin = label_binarize(y_true, classes=np.unique(y_true))
    n_classes = y_bin.shape[1]

    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC (OvR)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_pr_ovr(y_true: np.ndarray, scores: np.ndarray, class_names: List[str]):
    """
    One-vs-rest Precision-Recall curves. scores shape (n_samples, n_classes)
    """
    y_bin = label_binarize(y_true, classes=np.unique(y_true))
    n_classes = y_bin.shape[1]

    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], scores[:, i])
        ap = average_precision_score(y_bin[:, i], scores[:, i])
        ax.plot(recall, precision, linewidth=2, label=f"{class_names[i]} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (OvR)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_decision_boundary_2d(
    model: Pipeline,
    X_train_2d: np.ndarray,
    y_train: np.ndarray,
    X_test_2d: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    feature_labels: Tuple[str, str],
    show: str = "Predicted class",
    grid_step: float = 0.02,
    alpha_bg: float = 0.25,
):
    """
    show: "Predicted class" or "Max probability"
    """
    x_min, x_max = X_train_2d[:, 0].min() - 0.7, X_train_2d[:, 0].max() + 0.7
    y_min, y_max = X_train_2d[:, 1].min() - 0.7, X_train_2d[:, 1].max() + 0.7

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # predictions on grid
    if show == "Predicted class":
        Z = model.predict(grid).reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(6.3, 4.8))
        ax.contourf(xx, yy, Z, alpha=alpha_bg, levels=np.arange(len(class_names) + 1) - 0.5, cmap="viridis")
        ax.contour(xx, yy, Z, colors="k", linewidths=0.5, alpha=0.35)
    else:
        proba = _safe_predict_proba(model, grid)
        if proba is None:
            # fallback: try decision_function then softmax-like normalization
            dec = _safe_decision_function(model, grid)
            if dec is None:
                raise RuntimeError("No se puede obtener predict_proba/decision_function para la superficie.")
            if dec.ndim == 1:
                dec = np.vstack([-dec, dec]).T
            # Normalize to [0,1] approx (not a real probability)
            dec_shift = dec - dec.max(axis=1, keepdims=True)
            exp = np.exp(dec_shift)
            proba = exp / exp.sum(axis=1, keepdims=True)

        Z = proba.max(axis=1).reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(6.3, 4.8))
        c = ax.contourf(xx, yy, Z, alpha=0.9, cmap="magma")
        fig.colorbar(c, ax=ax, label="Max prob.")
        # also overlay boundaries from predicted class
        pred = proba.argmax(axis=1).reshape(xx.shape)
        ax.contour(xx, yy, pred, colors="white", linewidths=0.7, alpha=0.7)

    # scatter train/test
    palette = sns.color_palette("deep", n_colors=len(class_names))
    for cls in np.unique(y_train):
        ax.scatter(
            X_train_2d[y_train == cls, 0],
            X_train_2d[y_train == cls, 1],
            s=28,
            color=palette[int(cls)],
            edgecolor="k",
            linewidth=0.3,
            alpha=0.85,
            label=f"Train: {class_names[int(cls)]}",
            marker="o",
        )
    for cls in np.unique(y_test):
        ax.scatter(
            X_test_2d[y_test == cls, 0],
            X_test_2d[y_test == cls, 1],
            s=40,
            color=palette[int(cls)],
            edgecolor="k",
            linewidth=0.8,
            alpha=0.95,
            label=f"Test: {class_names[int(cls)]}",
            marker="^",
        )

    ax.set_xlabel(feature_labels[0])
    ax.set_ylabel(feature_labels[1])
    ax.set_title("Frontera de decisión (2D) - " + show)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def format_metric_value(x: float) -> str:
    return f"{x:.4f}"


# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="Iris Classifier Lab", layout="wide")
    set_matplotlib_style()

    st.title("Clasificación del Iris Dataset (Streamlit)")
    st.caption("Explora diferentes modelos, métricas, curvas ROC/PR y fronteras de decisión en 2D.")

    X_df, y_ser, feature_names, class_names = get_iris_dataframe()
    X = X_df.values
    y = y_ser.values

    with st.sidebar:
        st.header("Configuración")

        st.subheader("Datos")
        test_size = st.slider("Tamaño de test", min_value=0.1, max_value=0.5, value=0.25, step=0.05)
        stratify = st.checkbox("Stratify (recomendado)", value=True)
        random_state = st.number_input("Random state", min_value=0, max_value=10_000, value=RANDOM_STATE_DEFAULT, step=1)

        st.subheader("Preprocesamiento")
        scale = st.checkbox("Estandarizar (StandardScaler)", value=True)

        st.subheader("Modelo")
        model_name = st.selectbox(
            "Elige un modelo",
            ["Logistic Regression", "SVM (SVC)", "KNN", "Decision Tree", "Random Forest"],
        )

        st.markdown("---")
        st.subheader("Hiperparámetros")

        # Logistic Regression
        lr_C = st.slider("LR: C", 0.01, 10.0, 1.0, 0.01) if model_name == "Logistic Regression" else 1.0
        lr_max_iter = st.slider("LR: max_iter", 100, 2000, 500, 50) if model_name == "Logistic Regression" else 500

        # SVM
        svm_C = st.slider("SVM: C", 0.1, 50.0, 1.0, 0.1) if model_name == "SVM (SVC)" else 1.0
        svm_kernel = (
            st.selectbox("SVM: kernel", ["rbf", "linear", "poly", "sigmoid"])
            if model_name == "SVM (SVC)"
            else "rbf"
        )
        svm_gamma = (
            st.selectbox("SVM: gamma", ["scale", "auto"])
            if model_name == "SVM (SVC)" and svm_kernel in ["rbf", "poly", "sigmoid"]
            else "scale"
        )
        svm_degree = st.slider("SVM: degree (solo poly)", 2, 6, 3, 1) if (model_name == "SVM (SVC)" and svm_kernel == "poly") else 3

        # KNN
        knn_k = st.slider("KNN: k", 1, 25, 5, 1) if model_name == "KNN" else 5

        # Decision Tree
        dt_max_depth_ui = st.slider("DT: max_depth (0 = None)", 0, 15, 3, 1) if model_name == "Decision Tree" else 3
        dt_max_depth = None if (model_name == "Decision Tree" and dt_max_depth_ui == 0) else (dt_max_depth_ui if model_name == "Decision Tree" else None)

        # Random Forest
        rf_n_estimators = st.slider("RF: n_estimators", 50, 600, 200, 25) if model_name == "Random Forest" else 200
        rf_max_depth_ui = st.slider("RF: max_depth (0 = None)", 0, 25, 0, 1) if model_name == "Random Forest" else 0
        rf_max_depth = None if (model_name == "Random Forest" and rf_max_depth_ui == 0) else (rf_max_depth_ui if model_name == "Random Forest" else None)

        st.markdown("---")
        st.subheader("Evaluación")
        avg = st.selectbox("Promedio (multiclase)", ["macro", "weighted"], index=0)

        st.subheader("Visualización")
        show_cm = st.checkbox("Matriz de confusión", value=True)
        cm_norm = st.checkbox("Normalizar matriz de confusión", value=True)
        show_report = st.checkbox("Classification report", value=True)
        show_roc = st.checkbox("Curvas ROC (OvR)", value=True)
        show_pr = st.checkbox("Curvas Precision-Recall (OvR)", value=False)

        st.subheader("Frontera de decisión (2D)")
        show_boundary = st.checkbox("Mostrar frontera de decisión", value=True)
        feat_x = st.selectbox("Feature X", feature_names, index=0)
        feat_y = st.selectbox("Feature Y", feature_names, index=2)
        boundary_mode = st.selectbox("Qué mostrar", ["Predicted class", "Max probability"], index=0)
        grid_step = st.slider("Resolución grid (menor = más fino)", 0.01, 0.10, 0.03, 0.01)

    # Split
    stratify_y = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )

    # Model
    model = make_model(
        model_name,
        scale=scale,
        lr_C=lr_C,
        lr_max_iter=lr_max_iter,
        svm_C=svm_C,
        svm_kernel=svm_kernel,
        svm_gamma=svm_gamma,
        svm_degree=svm_degree,
        knn_k=knn_k,
        dt_max_depth=dt_max_depth,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, average=avg)

    # Layout
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", format_metric_value(metrics["accuracy"]))
    col2.metric("Precision", format_metric_value(metrics["precision"]))
    col3.metric("Recall", format_metric_value(metrics["recall"]))
    col4.metric("F1", format_metric_value(metrics["f1"]))

    st.markdown("---")

    left, right = st.columns([1.05, 1.0], gap="large")

    with left:
        st.subheader("Datos y predicciones")
        with st.expander("Ver dataset (primeras filas)", expanded=False):
            st.dataframe(X_df.assign(target=y_ser.map(lambda i: class_names[int(i)])).head(12), use_container_width=True)

        if show_cm:
            fig = plot_confusion_matrix(y_test, y_pred, class_names, normalize=cm_norm)
            st.pyplot(fig, clear_figure=True)

        if show_report:
            rep = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
            st.text(rep)

        if show_roc or show_pr:
            try:
                scores, kind = get_scores_for_curves(model, X_test)
                st.caption(f"Scores usados para curvas: {kind}")
                if show_roc:
                    fig = plot_roc_ovr(y_test, scores, class_names)
                    st.pyplot(fig, clear_figure=True)
                if show_pr:
                    fig = plot_pr_ovr(y_test, scores, class_names)
                    st.pyplot(fig, clear_figure=True)
            except Exception as e:
                st.warning(f"No se pudieron generar curvas ROC/PR: {e}")

    with right:
        st.subheader("Frontera de decisión (2D)")

        if show_boundary:
            # Build 2D dataset based on chosen features
            ix = feature_names.index(feat_x)
            iy = feature_names.index(feat_y)

            X_train_2d = X_train[:, [ix, iy]]
            X_test_2d = X_test[:, [ix, iy]]

            # Create and fit a *2D* model to match the selected features
            # (so the boundary is consistent)
            model_2d = make_model(
                model_name,
                scale=scale,
                lr_C=lr_C,
                lr_max_iter=lr_max_iter,
                svm_C=svm_C,
                svm_kernel=svm_kernel,
                svm_gamma=svm_gamma,
                svm_degree=svm_degree,
                knn_k=knn_k,
                dt_max_depth=dt_max_depth,
                rf_n_estimators=rf_n_estimators,
                rf_max_depth=rf_max_depth,
            )
            model_2d.fit(X_train_2d, y_train)

            try:
                fig = plot_decision_boundary_2d(
                    model_2d,
                    X_train_2d,
                    y_train,
                    X_test_2d,
                    y_test,
                    class_names,
                    feature_labels=(feat_x, feat_y),
                    show=boundary_mode,
                    grid_step=grid_step,
                )
                st.pyplot(fig, clear_figure=True)
            except Exception as e:
                st.warning(f"No se pudo dibujar la frontera de decisión: {e}")
        else:
            st.info("Activa 'Mostrar frontera de decisión' en la barra lateral.")

    st.markdown("---")
    st.subheader("Notas")
    st.write(
        "- La frontera de decisión se calcula en 2D entrenando el modelo solo con las 2 features seleccionadas.\n"
        "- Para ROC/PR en multiclase se usa esquema One-vs-Rest (OvR).\n"
        "- En SVM se activa `probability=True` para poder usar `predict_proba`."
    )


if __name__ == "__main__":
    main()
