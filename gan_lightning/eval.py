from gan_lightning.utils.noise import create_noise
import onnx
import onnxruntime
import torch.onnx
import numpy as np
import cv2
def eval_controllable(ckpt_path: str, num_samples: int = 25):



    from gan_lightning.src.models.classifiers.controllable_classifier import (
        Controllable_Classifier,
    )
    from gan_lightning.src.models.generators.deepconv_generator import (
        DeepConv_Generator,
    )
    
    classifier_ckpt_path, generator_ckpt_path = ckpt_path[0], ckpt_path[1]
    input_dim = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = Controllable_Classifier(mode="eval")
    classifier_state_dict = torch.load(classifier_ckpt_path)["state_dict"]
    classifier.load_state_dict(classifier_state_dict)
    
    generator = DeepConv_Generator(input_dim=input_dim).to(device)
    print(generator_ckpt_path)
    generator_state_dict = torch.load(generator_ckpt_path)["gen"]
    generator.load_state_dict(generator_state_dict)
    
    print("Models loaded successfully")


def eval_deepconv(onnx_path: str, input_dim:int=128):
    infer = onnxruntime.InferenceSession(onnx_path)
    input_name = infer.get_inputs()[0].name
    output_name = infer.get_outputs()[0].name
    noise = create_noise(1, input_dim).cpu().numpy()
    noise = noise.astype(np.float32)
    
    output = infer.run([output_name], {input_name: noise})
    output = np.transpose(output[0], (0, 2, 3, 1))
    cv2.imwrite("result.png", output[0] * 255)
    print("Inference completed")
    
def eval_simple(ckpt_path: str, num_samples: int = 25):
    pass


def eval_w(ckpt_path: str, num_samples: int = 25):
    pass


def eval_conditional(ckpt_path: str, num_samples: int = 25):
    pass
