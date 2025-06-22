import tensorflow as tf
from keras import layers, Model
from keras.applications.densenet import DenseNet201
from keras.applications.vgg19 import VGG19
from vit_keras import vit
import numpy as np
from scipy.stats import mode

def create_vision_transformer_model(num_classes, image_size=224):
    model = vit.vit_b16(
        image_size=image_size,
        activation='sigmoid',
        pretrained=True,
        include_top=True,
        pretrained_top=False,
        classes=num_classes
    )
    return model

def create_vgg19_model(num_classes, image_size=224):
    pre_trained_model = VGG19(input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet")
    x = layers.GlobalMaxPooling2D()(pre_trained_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(pre_trained_model.input, x)
    return model

def create_densenet201_model(num_classes, image_size=224):
    pre_trained_model = DenseNet201(input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet")
    x = layers.GlobalMaxPooling2D()(pre_trained_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(pre_trained_model.input, x)
    return model

def decision_fusion_predict(vit_model, vgg_model, densenet_model, test_generator, method='average', weights=None):
    predictions_1 = vit_model.predict(test_generator)
    predictions_2 = vgg_model.predict(test_generator)
    predictions_3 = densenet_model.predict(test_generator)

    if method == 'average':
        ensemble_predictions = (predictions_1 + predictions_2 + predictions_3) / 3
    elif method == 'weighted':
        if weights is None:
            weights = [0.4, 0.3, 0.3]
        ensemble_predictions = (predictions_1 * weights[0] + predictions_2 * weights[1] + predictions_3 * weights[2])
    elif method == 'majority':
        class_labels_1 = np.argmax(predictions_1, axis=1)
        class_labels_2 = np.argmax(predictions_2, axis=1)
        class_labels_3 = np.argmax(predictions_3, axis=1)
        ensemble_predictions, _ = mode([class_labels_1, class_labels_2, class_labels_3], axis=0)
        ensemble_predictions = np.squeeze(ensemble_predictions)
        return ensemble_predictions
    else:
        raise ValueError("Method must be 'average', 'weighted', or 'majority'")

    ensemble_predictions = np.argmax(ensemble_predictions, axis=1)
    return ensemble_predictions