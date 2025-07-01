import os
import warnings
import logging
import argparse
import numpy as np
import tensorflow as tf
from scipy.stats import mode
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

from models.fusion_model import create_fusion_model
from models.decision_fusion import create_vision_transformer_model, create_vgg19_model, create_densenet201_model
from train import compile_model, train_model_npy


# ==================== Suppress Warnings and Logs ==================== #
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


# ========================== Evaluation Utils ========================== #
def evaluate_fusion_model(model, X_test, y_test_original):
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)

    report = classification_report(y_test_original, predicted_labels)
    print("📊 Feature-Level Fusion - Classification Report:\n", report)

    cm = confusion_matrix(y_test_original, predicted_labels)
    print("📊 Feature-Level Fusion - Confusion Matrix:\n", cm)

    accuracy = accuracy_score(y_test_original, predicted_labels)
    precision = precision_score(y_test_original, predicted_labels, average=None)
    recall = recall_score(y_test_original, predicted_labels, average=None)
    f1 = f1_score(y_test_original, predicted_labels, average=None)

    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision per class: {precision}")
    print(f"✅ Recall per class: {recall}")
    print(f"✅ F1 Score per class: {f1}")
    print(f"✅ Balanced Accuracy: {np.mean(recall):.4f}")

    weighted_precision = sum(precision[i] * cm[i].sum() / cm.sum() for i in range(len(precision)))
    weighted_recall = sum(recall[i] * cm[i].sum() / cm.sum() for i in range(len(recall)))
    weighted_f1 = sum(f1[i] * cm[i].sum() / cm.sum() for i in range(len(f1)))

    print(f" Weighted Precision: {weighted_precision:.4f}")
    print(f" Weighted Recall: {weighted_recall:.4f}")
    print(f" Weighted F1 Score: {weighted_f1:.4f}")


def evaluate_decision_fusion(vit_model, vgg_model, densenet_model, X_test, y_test_original, method='majority', weights=None):
    preds_vit = vit_model.predict(X_test)
    preds_vgg = vgg_model.predict(X_test)
    preds_densenet = densenet_model.predict(X_test)

    if method == 'average':
        final_preds = (preds_vit + preds_vgg + preds_densenet) / 3
    elif method == 'weighted':
        if weights is None:
            raise ValueError("Weights must be provided for weighted fusion.")
        final_preds = preds_vit * weights[0] + preds_vgg * weights[1] + preds_densenet * weights[2]
    elif method == 'majority':
        labels = [np.argmax(p, axis=1) for p in [preds_vit, preds_vgg, preds_densenet]]
        final_preds, _ = mode(labels, axis=0)
        final_preds = np.squeeze(final_preds)
    else:
        raise ValueError(f"Invalid decision fusion method: {method}")

    if method != 'majority':
        final_preds = np.argmax(final_preds, axis=1)

    cm = confusion_matrix(y_test_original, final_preds)
    accuracy = accuracy_score(y_test_original, final_preds)
    precision = precision_score(y_test_original, final_preds, average=None, zero_division=0)
    recall = recall_score(y_test_original, final_preds, average=None, zero_division=0)
    f1 = f1_score(y_test_original, final_preds, average=None, zero_division=0)

    print(f"\n=== Decision-Level Fusion ({method.upper()}) ===")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision per class: {precision}")
    print(f"✅ Recall per class: {recall}")
    print(f"✅ F1 per class: {f1}")
    print(f"✅ Balanced Accuracy: {np.mean(recall):.4f}")
    print("📊 Confusion Matrix:\n", cm)

    weighted_precision = sum(precision[i] * cm[i].sum() / cm.sum() for i in range(len(precision)))
    weighted_recall = sum(recall[i] * cm[i].sum() / cm.sum() for i in range(len(recall)))
    weighted_f1 = sum(f1[i] * cm[i].sum() / cm.sum() for i in range(len(f1)))

    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f" Weighted Recall: {weighted_recall:.4f}")
    print(f" Weighted F1 Score: {weighted_f1:.4f}")


# ========================== Argument Parser ========================== #
def parse_arguments():
    parser = argparse.ArgumentParser(description="Fusion-Based Dermatology Classification")

    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('--fusion', type=str, choices=['feature', 'decision'], default='feature')
    parser.add_argument('--decision_method', type=str, choices=['average', 'weighted', 'majority'], default='majority')

    parser.add_argument('--train_npy', type=str, required=True)
    parser.add_argument('--train_npy_labels', type=str, required=True)
    parser.add_argument('--test_npy', type=str, required=True)
    parser.add_argument('--test_npy_labels', type=str, required=True)

    parser.add_argument('--feature_checkpoint', type=str)
    parser.add_argument('--vit_checkpoint', type=str)
    parser.add_argument('--vgg_checkpoint', type=str)
    parser.add_argument('--densenet_checkpoint', type=str)

    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--n_class', type=int, default=3)

    return parser.parse_args()


# ========================== Main Function ========================== #
def main():
    args = parse_arguments()
    tf.random.set_seed(42)

    X_train = np.load(args.train_npy)
    y_train = np.load(args.train_npy_labels)
    X_test = np.load(args.test_npy)
    y_test = np.load(args.test_npy_labels)

    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    if args.fusion == 'feature':
        print(" Feature-Level Fusion Mode Selected")
        model = create_fusion_model(train_dir=None, image_size=args.image_size, num_classes=args.n_class)
        model = compile_model(model)

        if args.mode == 'train':
            print(" Training Feature-Level Fusion Model...")
            train_model_npy(model, X_train, y_train_cat, X_test, y_test_cat,
                            batch_size=args.batch_size, checkpoint_path=args.feature_checkpoint, epochs=args.epochs)
        else:
            print(" Evaluating Feature-Level Fusion Model...")
            model.load_weights(args.feature_checkpoint)
            evaluate_fusion_model(model, X_test, y_test)

    elif args.fusion == 'decision':
        print(f" Decision-Level Fusion Mode ({args.decision_method}) Selected")

        vit_model = create_vision_transformer_model(args.n_class, image_size=args.image_size)
        vgg_model = create_vgg19_model(args.n_class, image_size=args.image_size)
        densenet_model = create_densenet201_model(args.n_class, image_size=args.image_size)

        vit_model = compile_model(vit_model)
        vgg_model = compile_model(vgg_model)
        densenet_model = compile_model(densenet_model)

        if args.mode == 'train':
            print("Training Individual Models...")
            train_model_npy(vit_model, X_train, y_train_cat, X_test, y_test_cat,
                            batch_size=args.batch_size, checkpoint_path=args.vit_checkpoint, epochs=args.epochs)
            train_model_npy(vgg_model, X_train, y_train_cat, X_test, y_test_cat,
                            batch_size=args.batch_size, checkpoint_path=args.vgg_checkpoint, epochs=args.epochs)
            train_model_npy(densenet_model, X_train, y_train_cat, X_test, y_test_cat,
                            batch_size=args.batch_size, checkpoint_path=args.densenet_checkpoint, epochs=args.epochs)
        else:
            print("Evaluating Decision-Level Fusion Model...")
            vit_model.load_weights(args.vit_checkpoint)
            vgg_model.load_weights(args.vgg_checkpoint)
            densenet_model.load_weights(args.densenet_checkpoint)

            weights = [0.4, 0.3, 0.3] if args.decision_method == 'weighted' else None
            evaluate_decision_fusion(vit_model, vgg_model, densenet_model,
                                     X_test, y_test, method=args.decision_method, weights=weights)


if __name__ == "__main__":
    main()
