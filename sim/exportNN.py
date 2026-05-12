import torch
import matplotlib.pyplot as plt
from algorithms import NN_Controller
from parameters import params
from mlp_model import MLPModel

if __name__ == "__main__":
    ## CONFIG:
    
    # policy_type = "MLP" #GRU or MLP
    # model_path = f"/{policy_type}/policy_weights_1000cars_preTrain.pth"
    
    # policy_type = "GRU"
    # model_path = f"{policy_type}/policy_weights_GRU_100_cars.pth"
    
    policy_type = "MLP"
    model_path = f"{policy_type}/policy_weights_1000cars_preTrain.pth"
    success = False
    
    
    nn_controller = NN_Controller(type=policy_type, params=params)
    nn_controller.load_weigths(f"sim/NN_controllers/{model_path}")
    
    onnx_path = f"sim/export_control_models/{model_path}.onnx"
    nn_controller.policy.eval()

    if policy_type == "GRU":
        dummy_input = torch.randn(1, 1, 8)   # batch, seq_len, input_size
        dummy_h0 = torch.randn(1, 1, 128)
        torch.onnx.export(
            nn_controller.policy,
            (dummy_input, dummy_h0),
            onnx_path,
            input_names=["policy_input", "h0"],
            output_names=["control_output"],
            opset_version=17,
            dynamo=True,
        )
        success = True
    elif policy_type == "MLP":
        dummy_input = torch.randn(1, 8, dtype=torch.float32)
        torch.onnx.export(
            nn_controller.policy,
            dummy_input,
            onnx_path,
            input_names=["policy_input"],
            output_names=["control_output"],
            dynamic_axes={
                "policy_input": {0: "batch"},
                "control_output": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
    )
        success = True
    else:
        raise ValueError("[export_NN] Not valid policy type")
    
    if success == True:
        print(f"{policy_type} model exported to ONNX model at path {onnx_path}")