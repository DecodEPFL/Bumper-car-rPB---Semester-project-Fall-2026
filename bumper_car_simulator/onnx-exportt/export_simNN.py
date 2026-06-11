import torch
import numpy as np
from mlp_model import MLPModel
from parameters import params

if __name__ == "__main__":
    init_state = np.array([5.0, 2.0, 0, 3.0, 0.1, 0.3, 0.5], dtype=np.float32)

    mlp_model = MLPModel(
        initial_state=init_state,
        params=params,
        model_path="best_models/model_kinematic_mlp.pth",
        input_dim=2,
        state_dim=4,
        output_dim=3,
        hidden_sizes=[256, 128]
    )

    model = mlp_model.model
    model.eval()

    onnx_path = "export_mlp_sim/model_kinematic_mlp.onnx"

    dummy_input = torch.randn(1, 6, dtype=torch.float32)
    # 6 because input_dim + state_dim = 2 + 4

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["model_input"],
        output_names=["model_output"],
        dynamic_axes={
            "model_input": {0: "batch"},
            "model_output": {0: "batch"},
        },
        opset_version=17,
    )

    print(f"Exported ONNX model to {onnx_path}")