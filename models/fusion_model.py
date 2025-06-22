import tensorflow as tf
from keras import layers, Model
from keras.applications.densenet import DenseNet201
from keras.applications.vgg19 import VGG19
from vit_keras import vit
import os

def create_vision_transformer_model(image_size=224):
    model = vit.vit_b16(
        image_size=image_size,
        activation='sigmoid',
        pretrained=True,
        include_top=False,
        pretrained_top=False,
    )
    return model

def create_fusion_model(train_dir, image_size=224, num_classes=None):
    if num_classes is None:
        num_classes = len(os.listdir(train_dir))

    vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    densenet_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    vit_model = create_vision_transformer_model(image_size=image_size)

    input_layer = tf.keras.Input(shape=(image_size, image_size, 3))

    vit_features = vit_model(input_layer)
    vgg_features = vgg_model(input_layer)
    densenet_features = densenet_model(input_layer)

    vgg_globalAvgPool = layers.GlobalMaxPooling2D()(vgg_features)
    vgg_dense_layer = layers.Dense(768, activation='relu')(vgg_globalAvgPool)

    densenet_globalAvgPool = layers.GlobalMaxPooling2D()(densenet_features)
    densenet_dense_layer = layers.Dense(768, activation='relu')(densenet_globalAvgPool)

    vit_flattened = layers.Flatten()(vit_features)
    vgg_flattened = layers.Flatten()(vgg_dense_layer)
    densenet_flattened = layers.Flatten()(densenet_dense_layer)

    concatenated_features = densenet_flattened + vgg_flattened + vit_flattened

    concatenated_features = layers.Dense(512, activation='relu')(concatenated_features)
    concatenated_features = layers.Dropout(0.5)(concatenated_features)
    output = layers.Dense(num_classes, activation='softmax')(concatenated_features)

    model = Model(inputs=input_layer, outputs=output)
    return model