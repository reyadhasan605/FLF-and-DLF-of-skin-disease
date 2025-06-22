import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def evaluate_model(model, test_generator):
    predictions = model.predict(test_generator)
    final_predictions = np.argmax(predictions, axis=1)
    true_labels = test_generator.classes

    # Classification report
    report = classification_report(true_labels, final_predictions)
    print("Feature-Level Fusion Classification Report:\n", report)

    # Confusion matrix
    cm = confusion_matrix(true_labels, final_predictions)
    print("Feature-Level Fusion Confusion Matrix:\n", cm)

    # Per-class metrics
    accuracy = accuracy_score(true_labels, final_predictions)
    precision = precision_score(true_labels, final_predictions, average=None)
    recall = recall_score(true_labels, final_predictions, average=None)
    f1 = f1_score(true_labels, final_predictions, average=None)

    print("Feature-Level Fusion Accuracy:", accuracy)
    print("Feature-Level Fusion Precision per class:", precision)
    print("Feature-Level Fusion Recall per class:", recall)
    print("Feature-Level Fusion F1 Score per class:", f1)

    # Weighted averages
    weighted_precision = sum(precision[i] * cm[i].sum() / cm.sum() for i in range(len(precision)))
    weighted_recall = sum(recall[i] * cm[i].sum() / cm.sum() for i in range(len(recall)))
    weighted_f1 = sum(f1[i] * cm[i].sum() / cm.sum() for i in range(len(f1)))

    print("Feature-Level Fusion Balanced Accuracy:", np.mean(recall))
    print("Feature-Level Fusion Weighted Avg Precision:", weighted_precision)
    print("Feature-Level Fusion Weighted Avg Recall:", weighted_recall)
    print("Feature-Level Fusion Weighted Avg F1:", weighted_f1)

    return report, cm, accuracy, precision, recall, f1

def evaluate_ensemble(ensemble_predictions, true_labels):
    # Classification report
    report = classification_report(true_labels, ensemble_predictions)
    print("Decision-Level Fusion Classification Report:\n", report)

    # Confusion matrix
    cm = confusion_matrix(true_labels, ensemble_predictions)
    print("Decision-Level Fusion Confusion Matrix:\n", cm)

    # Per-class metrics
    accuracy = accuracy_score(true_labels, ensemble_predictions)
    precision = precision_score(true_labels, ensemble_predictions, average=None)
    recall = recall_score(true_labels, ensemble_predictions, average=None)
    f1 = f1_score(true_labels, ensemble_predictions, average=None)

    print("Decision-Level Fusion Accuracy:", accuracy)
    print("Decision-Level Fusion Class-wise Precision:", precision)
    print("Decision-Level Fusion Class-wise Recall:", recall)
    print("Decision-Level Fusion Class-wise F1:", f1)

    # Weighted averages
    weighted_precision = sum(precision[i] * cm[i].sum() / cm.sum() for i in range(len(precision)))
    weighted_recall = sum(recall[i] * cm[i].sum() / cm.sum() for i in range(len(recall)))
    weighted_f1 = sum(f1[i] * cm[i].sum() / cm.sum() for i in range(len(f1)))

    print("Decision-Level Fusion Balanced Accuracy:", np.mean(recall))
    print("Decision-Level Fusion Weighted Avg Precision:", weighted_precision)
    print("Decision-Level Fusion Weighted Avg Recall:", weighted_recall)
    print("Decision-Level Fusion Weighted Avg F1:", weighted_f1)

    return report, cm, accuracy, precision, recall, f1