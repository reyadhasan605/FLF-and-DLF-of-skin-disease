import tensorflow as tf
from data.dataloader import get_data_generators
from models.fusion_model import create_fusion_model
from models.decision_fusion import create_vision_transformer_model, create_vgg19_model, create_densenet201_model
from train import compile_model, train_model
from evaluate import evaluate_feature_fusion, evaluate_decision_fusion
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Feature and Decision-Level Fusion for Dermatological Diagnosis')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='Mode to run: "train" or "test" (default: test)')
    parser.add_argument('--fusion', type=str, choices=['feature', 'decision'], default='decision',
                        help='Fusion type: "feature" for feature-level fusion or "decision" for decision-level fusion (default: feature)')
    parser.add_argument('--decision_method', type=str, choices=['average', 'weighted', 'majority'], default='average',
                        help='Decision-level fusion method: "average", "weighted", or "majority" (default: average)')
    parser.add_argument('--train_dir', type=str,
                        default="",
                        help='Path to training data directory')
    parser.add_argument('--test_dir', type=str,
                        default="",
                        help='Path to test data directory')
    parser.add_argument('--feature_checkpoint', type=str,
                        default="",
                        help='Path to feature-level fusion model checkpoint')
    parser.add_argument('--vit_checkpoint', type=str,
                        default="",
                        help='Path to ViT model checkpoint')
    parser.add_argument('--vgg_checkpoint', type=str,
                        default="",
                        help='Path to VGG19 model checkpoint')
    parser.add_argument('--densenet_checkpoint', type=str,
                        default="",
                        help='Path to DenseNet201 model checkpoint')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size (height and width) in pixels (default: 224)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for data generators (default: 16)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs (default: 150)')
    return parser.parse_args()


def main():
    args = parse_arguments()
    tf.random.set_seed(42)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


    # Load data
    train_generator, test_generator = get_data_generators(
        args.train_dir, args.test_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size
    )

    # Number of classes
    num_classes = len(os.listdir(args.train_dir))

    if args.fusion == 'feature':
        print("=== Feature-Level Fusion ===")
        feature_model = create_fusion_model(args.train_dir, image_size=args.image_size)
        feature_model = compile_model(feature_model)

        if args.mode == 'train':
            print("Training Feature-Level Fusion Model...")
            train_model(feature_model, train_generator, test_generator,
                        args.feature_checkpoint, epochs=args.epochs)

        else:
            print("Testing Feature-Level Fusion Model...")
            feature_model.load_weights(args.feature_checkpoint)
            evaluate_feature_fusion(feature_model, test_generator)

    elif args.fusion == 'decision':
        print(f"=== Decision-Level Fusion ({args.decision_method.capitalize()} Method) ===")
        vit_model = create_vision_transformer_model(num_classes, image_size=args.image_size)
        vgg_model = create_vgg19_model(num_classes, image_size=args.image_size)
        densenet_model = create_densenet201_model(num_classes, image_size=args.image_size)

        vit_model = compile_model(vit_model)
        vgg_model = compile_model(vgg_model)
        densenet_model = compile_model(densenet_model)

        if args.mode == 'train':
            print("Training Individual Models for Decision-Level Fusion...")
            train_model(vit_model, train_generator, test_generator,
                        args.vit_checkpoint, epochs=args.epochs)
            train_model(vgg_model, train_generator, test_generator,
                        args.vgg_checkpoint, epochs=args.epochs)
            train_model(densenet_model, train_generator, test_generator,
                        args.densenet_checkpoint, epochs=args.epochs)
        else:
            print("Testing Decision-Level Fusion Model...")
            vit_model.load_weights(args.vit_checkpoint)
            vgg_model.load_weights(args.vgg_checkpoint)
            densenet_model.load_weights(args.densenet_checkpoint)
            weights = [0.4, 0.3, 0.3] if args.decision_method == 'weighted' else None
            evaluate_decision_fusion(vit_model, vgg_model, densenet_model,
                                     test_generator, method=args.decision_method, weights=weights)


if __name__ == "__main__":
    main()