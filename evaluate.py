from utils.metrics import evaluate_model, evaluate_ensemble
from models.decision_fusion import decision_fusion_predict

def evaluate_feature_fusion(model, test_generator):
    report, cm, accuracy, precision, recall, f1 = evaluate_model(model, test_generator)
    return report, cm, accuracy, precision, recall, f1

def evaluate_decision_fusion(vit_model, vgg_model, densenet_model, test_generator, method='average', weights=None):
    ensemble_predictions = decision_fusion_predict(vit_model, vgg_model, densenet_model, test_generator, method, weights)
    true_labels = test_generator.classes
    report, cm, accuracy, precision, recall, f1 = evaluate_ensemble(ensemble_predictions, true_labels)
    return report, cm, accuracy, precision, recall, f1