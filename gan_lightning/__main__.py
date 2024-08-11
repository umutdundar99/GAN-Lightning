from gan_lightning.train import GAN_Lightning
from gan_lightning.eval import (
    eval_controllable,
    eval_deepconv,
    eval_simple,
    eval_w,
    eval_conditional,
)
from gan_lightning.export import export_model

if __name__ == "__main__":

    import argparse
    from hydra import compose, initialize

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "ConditionalGAN",
            "Controllable_Classifier",
            "DeepConvGAN",
            "SimpleGAN",
            "WGAN",
        ],
        default="SimpleGAN",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "export"],
        default="train",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        nargs="+",
        help="Path to the checkpoint file for export mode",
    )

    parser.add_argument(
        "--input-dim",
        type=int,
        default=128,
        help="Dimension of the noise vector for export and eval mode",
    )

    parser.add_argument(
        "--img-channel",
        type=int,
        default=1,
        help="Number of image channels for export and eval mode",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=28,
        help="Size of the input image for export and eval mode",
    )

    parser.add_argument(
        "--onnx-path",
        type=str,
        default=None,
        help="Path to the onnx file for evaluation mode, if the model is Controllable_Classifier, firstly provide the path to the classifier onnx and then the generator onnx file path, provide only one path for the rest of the models",
    )

    args = parser.parse_args()

    if args.mode == "eval":
        if args.model == "Controllable_Classifier":
            eval_controllable(args.onnx_path, args.input_dim)
        elif args.model == "DeepConvGAN":
            eval_deepconv(args.onnx_path, args.input_dim)
        elif args.model == "SimpleGAN":
            eval_simple(args.onnx_path, args.input_dim)
        elif args.model == "WGAN":
            eval_w(args.onnx_path, args.input_dim)
        elif args.model == "ConditionalGAN":
            eval_conditional(args.onnx_path, args.input_dim)

    elif args.mode == "train":

        initialize(config_path="src/config/training_configs", version_base=None)
        config_name = f"{args.model}"
        cfg = compose(config_name=config_name)
        GAN_Lightning(cfg)

    elif args.mode == "export":

        export_model(
            args.model,
            args.ckpt_path[0],
            input_dim=args.input_dim,
            img_channel=args.img_channel,
            input_size=args.input_size,
        )
