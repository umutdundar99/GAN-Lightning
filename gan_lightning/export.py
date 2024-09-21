import onnx
import torch
import torch.onnx
from onnxsim import simplify
from gan_lightning.src.models import registered_models
from typing import Optional


def export_model(
    model: Optional[str],
    ckpt_path: Optional[str],
    input_dim: Optional[int],
    img_channel: Optional[int],
    input_size: Optional[int],
    num_classes: Optional[int] = 40,
    dataset_name: str = "mnist",
    save_dir: str = "gan_lightning/onnxes",
):
    input_dim = input_dim + num_classes if num_classes is not None else input_dim
    export_kwargs = {
        "input_dim": input_dim,
        "img_channel": img_channel,
        "input_size": input_size[0] if len(input_size) > 1 else input_size,
        "num_classes": num_classes,
    }

    model = registered_models[model]
    model = model(mode="eval", **export_kwargs)
    model_name = model.name()
    state_dict = torch.load(ckpt_path)["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    if input_dim is not None:
        dummy_input = torch.randn(1, input_dim)
    else:
        dummy_input = torch.randn(1, img_channel, *input_size)

    torch.onnx.export(
        model,
        dummy_input,
        f"{save_dir}/{dataset_name}-{model_name}.onnx",
        input_names=["input"],
        output_names=["output"],
        verbose=False,
    )

    # simplify onnx model
    onnx_model = onnx.load(f"{save_dir}/{dataset_name}-{model_name}.onnx")
    simplified_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(simplified_model, f"{save_dir}/{dataset_name}-{model_name}.onnx")
