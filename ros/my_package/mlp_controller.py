import numpy as np
import onnxruntime as ort
from rclpy.node import Node
import torch
from .gruPolicy import PolicyGRU
from .parameters import CarParams as params

class NNController:
    def __init__(self, model_path, model_type, device="cpu"):
        self.model_type = model_type
        if model_type == "MLP":
            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        
        elif model_type == "GRU":        
            self.gru_hidden = None
            self.nn_controller = PolicyGRU(params)
            self.nn_controller.load_state_dict(torch.load(
                "/home/ubuntubirger/ros2_ws/src/my_package/GRU_control_models/pth/policy_weights_GRU_50_cars.pth"), strict=False)

        
        

    def build_policy_input(self, car_state, car_state_final):
        car_state = np.asarray(car_state, dtype=np.float32).reshape(-1)
        car_state_final = np.asarray(car_state_final, dtype=np.float32).reshape(-1)

        if car_state.shape != car_state_final.shape:
            raise ValueError(f"Shape mismatch: {car_state.shape} vs {car_state_final.shape}")

        if car_state.ndim != 1:
            raise ValueError(f"Expected 1D state vector, got shape {car_state.shape}")

        error = car_state_final - car_state

        dx_i = error[0]
        dy_i = error[1]
        theta = car_state[2]

        dx_local = np.cos(theta) * dx_i + np.sin(theta) * dy_i
        dy_local = -np.sin(theta) * dx_i + np.cos(theta) * dy_i

        sin_dtheta = np.sin(error[2])
        cos_dtheta = np.cos(error[2])

        error_body = np.concatenate([
            np.array([dx_local, dy_local, sin_dtheta, cos_dtheta], dtype=np.float32),
            error[3:].astype(np.float32)
        ], axis=0)

        # shape should be (8,)
        if error_body.shape != (8,):
            raise ValueError(f"Expected policy input shape (8,), got {error_body.shape}")

        return error_body.reshape(1, 1, 8).astype(np.float32)

    # For MLP
    # def compute_control(self, car_state, car_state_final, node):
    #     policy_input = self.build_policy_input(car_state, car_state_final)

    #     output = self.session.run(
    #         [self.output_name],
    #         {self.input_name: policy_input}
    #     )[0]
    #     node.get_logger().info(f"policy-input: {policy_input}")

    #     return output.squeeze(0)
    
    def compute_control(self, car_state, car_state_final, node):
        # TEST GRU model
        if self.model_type == "GRU":
            if self.gru_hidden is not None:
                self.gru_hidden = self.gru_hidden.detach()

            with torch.no_grad():
                policy_input = self.build_policy_input(car_state, car_state_final)

                u, self.gru_hidden = self.nn_controller(
                    policy_input,
                    h0=self.gru_hidden,
                    return_hidden=True,
                )

                safe_input = u[0].detach().cpu().numpy()
        
        elif self.model_type == "MLP":
            policy_input = self.build_policy_input(car_state, car_state_final)
            policy_input = policy_input.reshape(1, 8).astype(np.float32)
            output = self.session.run(
                [self.output_name],
                {
                    "policy_input": policy_input,
                }
            )[0]
            safe_input = output.squeeze(0)
            

        return safe_input