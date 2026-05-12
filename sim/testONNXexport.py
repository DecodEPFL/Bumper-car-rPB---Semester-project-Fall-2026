import os
import numpy as np
import torch
import onnxruntime as ort
from algorithms import NN_Controller
from parameters import CarParams as params



def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


if __name__ == "__main__":
    # CONFIG
    policy_type = "GRU"
    model_path = f"{policy_type}/policy_weights_GRU_50_cars.pth"
    onnx_path = f"sim/export_mlp_control_models/{os.path.splitext(model_path)[0]}.onnx"
    # Load PyTorch model
    controller = NN_Controller(type=policy_type, params=params)
    controller.load_weigths(f"sim/NN_controllers/{model_path}")
    controller.policy.eval()

    # Match the GRU dimensions from the model
    num_layers = controller.policy.gru.num_layers
    num_directions = 2 if controller.policy.gru.bidirectional else 1
    hidden_size = controller.policy.gru.hidden_size

    # Test input
    batch_size = 1
    seq_len = 1
    input_size = 8

    x = torch.randn(batch_size, seq_len, input_size, dtype=torch.float32)
    h0 = torch.randn(
        num_layers * num_directions,
        batch_size,
        hidden_size,
        dtype=torch.float32
    )

    # PyTorch inference
    with torch.no_grad():
        torch_out = controller.policy(x, h0)

    torch_out_np = to_numpy(torch_out)

    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    ort_inputs = {
        "policy_input": to_numpy(x),
        "h0": to_numpy(h0),
    }

    ort_outs = session.run(None, ort_inputs)
    onnx_out_np = ort_outs[0]

    # Compare
    abs_diff = np.max(np.abs(torch_out_np - onnx_out_np))
    close = np.allclose(torch_out_np, onnx_out_np, rtol=1e-4, atol=1e-5)

    print("PyTorch output shape:", torch_out_np.shape)
    print("ONNX output shape:   ", onnx_out_np.shape)
    print("Max absolute diff:   ", abs_diff)
    print("Allclose:            ", close)

    print("\nPyTorch output:")
    print(torch_out_np)

    print("\nONNX output:")
    print(onnx_out_np)