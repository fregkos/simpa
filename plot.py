import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
)
import seaborn as sns

def plot_confusion_matrix(y, y_pred):
    class_names = ["cs", "econ", "eess", "math", "physics", "q-bio", "q-fin", "stat"]
    f, axes = plt.subplots(2, 4, figsize=(15, 7), dpi=600)
    axes = axes.ravel()

    for i in range(8):
        cm = confusion_matrix(y[:, i], y_pred[:, i])
        disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
        disp.plot(ax=axes[i], values_format=".4g")
        disp.ax_.set_title(f"Class: {class_names[i]}")
        if i < 4:
            disp.ax_.set_xlabel("")
        if i % 4 != 0:
            disp.ax_.set_ylabel("")

        disp.im_.colorbar.remove()

    plt.subplots_adjust(wspace=0.15, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
    plt.show()
    plt.savefig("confusion_matrix.png", bbox_inches="tight")


def plot_micro_average_precision(y, y_pred):
    class_names = ["cs", "econ", "eess", "math", "physics", "q-bio", "q-fin", "stat"]

    # Compute precision-recall and average precision for each class
    precision_recall_curves = []
    average_precisions = []

    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(
            y[:, i], y_pred[:, i]
        )
        ap = average_precision_score(y[:, i], y_pred[:, i])
        precision_recall_curves.append((precision, recall))
        average_precisions.append(ap)

    # Compute micro-average precision-recall curve
    true_labels_flat = y.ravel()
    predicted_probs_flat = y_pred.ravel()
    micro_precision, micro_recall, _ = precision_recall_curve(
        true_labels_flat, predicted_probs_flat
    )
    micro_average_precision = average_precision_score(
        true_labels_flat, predicted_probs_flat
    )

    # Plot Precision-Recall Curves with Seaborn
    plt.figure(figsize=(10, 8), dpi=600)
    sns.set_theme(style="whitegrid")

    # Plot micro-average curve
    plt.plot(
        micro_recall,
        micro_precision,
        label=f"Micro-average precision-recall (AP = {micro_average_precision:.2f})",
        color="gold",
    )

    # Colors for classes
    palette = sns.color_palette("tab10", len(class_names))

    # Plot each class curve
    for i, (precision, recall) in enumerate(precision_recall_curves):
        plt.plot(
            recall,
            precision,
            label=f"Class {class_names[i]} (AP = {average_precisions[i]:.2f})",
            color=palette[i],
        )

    # Customize the plot
    plt.title(
        "Micro Precision-Recall Curve for Multi-Label Classification", fontsize=16
    )
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.legend(loc="best", fontsize="small", title="Legend")
    plt.tight_layout()
    plt.savefig("micro_precision_recall_curve.png")
