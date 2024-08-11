import onnx
import torch
import torch.onnx
from onnxsim import simplify
from gan_lightning.src.models import registered_models



def export_model(model:str, ckpt_path:str, input_dim:int, img_channel:int, input_size:int):
    
    export_kwargs = {"input_dim":input_dim,
                    "img_channel":img_channel,
                    "input_size": input_size}

    model = registered_models[model]
    model = model(mode="eval", **export_kwargs)
    state_dict = torch.load(ckpt_path)["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    dummy_input = torch.randn(1, input_dim)
    torch.onnx.export(model, dummy_input, f"{model.get_name()}.onnx", input_names=["input"], output_names=["output"])
    
    # simplify onnx model
    onnx_model = onnx.load(f"{model.get_name()}.onnx")
    simplified_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(simplified_model, f"{model.get_name()}_simplified.onnx")
