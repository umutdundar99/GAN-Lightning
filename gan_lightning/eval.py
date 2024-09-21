import cv2
import os
import matplotlib.pyplot as plt
import torch
import onnxruntime
import numpy as np


def create_noise(num_samples: int, input_dim: int, seed):
    torch.random.manual_seed(seed)
    noise = torch.randn(num_samples, input_dim)
    return noise


def updated_noise(noise, weight):
    new_noise = noise + (noise.grad * weight)
    return new_noise


def save_images(image: np.ndarray, path: str, step: int):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    image = (image + 1) / 2
    image = image.detach().cpu().numpy()
    images = [image[i].transpose(1, 2, 0) * 255 for i in range(3)]

    for i in range(3):
        os.makedirs(os.path.join(path, str(i)), exist_ok=True)
        cv2.imwrite(os.path.join(path, str(i), f"{step}.png"), images[i])


def eval_controllable(**kwargs: dict):
    ckpt_paths = kwargs["ckpt_path"]
    num_classes = kwargs["num_classes"]
    input_dim = kwargs["input_dim"]
    classifier_ckpt_path, generator_ckpt_path = ckpt_paths[0], ckpt_paths[1]
    input_size = kwargs["input_size"]
    img_channel = kwargs["img_channel"]
    steps = 20

    from gan_lightning.src.models.classifiers.controllable_classifier import (
        Controllable_Classifier as Classifier,
    )
    from gan_lightning.src.models.gans.DeepConvGAN import DeepConvGAN as Generator

    save_path = "eval_results"

    os.makedirs(save_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = Classifier(mode="eval", num_classes=num_classes).to(device)
    classifier_state_dict = torch.load(classifier_ckpt_path)["state_dict"]
    classifier.load_state_dict(classifier_state_dict, strict=False)
    classifier.eval()

    generator = Generator(
        input_dim=input_dim, img_channel=img_channel, input_size=input_size, mode="eval"
    ).to(device)
    generator_state_dict = torch.load(generator_ckpt_path)["state_dict"]
    generator.load_state_dict(generator_state_dict, strict=False)
    generator.eval()

    print("Models loaded successfully")

    #     feature_names = ["5oClockShadow", "ArchedEyebrows", "Attractive", "BagsUnderEyes", "Bald", "Bangs",
    # "BigLips", "BigNose", "BlackHair", "BlondHair", "Blurry", "BrownHair", "BushyEyebrows", "Chubby",
    # "DoubleChin", "Eyeglasses", "Goatee", "GrayHair", "HeavyMakeup", "HighCheekbones", "Male",
    # "MouthSlightlyOpen", "Mustache", "NarrowEyes", "NoBeard", "OvalFace", "PaleSkin", "PointyNose",
    # "RecedingHairline", "RosyCheeks", "Sideburn", "Smiling", "StraightHair", "WavyHair", "WearingEarrings",
    # "WearingHat", "WearingLipstick", "WearingNecklace", "WearingNecktie", "Young"]
    feature_names = ["Smiling"]

    for feature in feature_names:
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
        noise = create_noise(5, input_dim, device).requires_grad_()
        indice = feature_names.index(feature)
        os.makedirs(os.path.join(save_path, feature), exist_ok=True)
        for i in range(steps):
            optimizer.zero_grad()
            generated_image = generator(noise)
            fake_score = classifier(generated_image)[:, indice].mean()
            print(f"Step: {i}, Score: {fake_score.item()}")
            fake_score.backward()
            noise.data = updated_noise(noise, 1 / steps)
            save_images(generated_image, os.path.join(save_path, feature), i)


def eval_deepconv(onnx_path: str, input_dim: int = 128, **kwargs: dict):
    save_path = "eval_results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    infer = onnxruntime.InferenceSession(onnx_path[0])
    input_name = infer.get_inputs()[0].name
    output_name = infer.get_outputs()[0].name

    # infer 20 images and save in a plot
    noise = create_noise(20, input_dim, 42).cpu().numpy()
    noise = noise.astype(np.float32)
    _, ax = plt.subplots(4, 5, figsize=(20, 20))
    seed = 0
    for i in range(4):
        for j in range(5):
            _noise = noise[seed]
            _noise = _noise.reshape(1, input_dim)
            output = infer.run([output_name], {input_name: _noise})
            output = np.transpose(output[0], (0, 2, 3, 1))
            output = (output + 1) / 2
            ax[i, j].imshow(output[0])
            ax[i, j].axis("off")
            seed += 1
    plt.savefig(f"eval_results/{onnx_path[0].split('/')[-1].split('.')[0]}.png")


def eval_simple(ckpt_path: str, num_samples: int = 25):
    pass


def eval_w(ckpt_path: str, num_samples: int = 25):
    pass


def eval_conditional(
    onnx_path: str, input_dim: int = 128, number_to_generate: int = 5, **kwargs: dict
):
    # number must be between 0-9
    assert number_to_generate in range(10)
    save_path = "eval_results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    infer = onnxruntime.InferenceSession(onnx_path[0])
    input_name = infer.get_inputs()[0].name
    output_name = infer.get_outputs()[0].name

    # infer 20 images and save in a plot
    noise = create_noise(20, input_dim, 42).cpu().numpy()
    # create one hot labels for 8
    one_hot_labels = np.eye(10)[np.array([number_to_generate] * 20)]
    noise = np.concatenate((noise, one_hot_labels), axis=1)
    input_dim += kwargs["num_classes"]
    noise = noise.astype(np.float32)
    _, ax = plt.subplots(4, 5, figsize=(20, 20))
    seed = 0
    for i in range(4):
        for j in range(5):
            _noise = noise[seed]
            _noise = _noise.reshape(1, input_dim)
            output = infer.run([output_name], {input_name: _noise})
            output = np.transpose(output[0], (0, 2, 3, 1))
            output = (output + 1) / 2
            ax[i, j].imshow(output[0])
            ax[i, j].axis("off")
            seed += 1
    plt.savefig(f"eval_results/{onnx_path[0].split('/')[-1].split('.')[0]}.png")
