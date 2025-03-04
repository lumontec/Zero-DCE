import torch
import model  # Importing your model

def export_to_onnx():
    # Load the model
    DCE_net = model.enhance_net_nopool()
    DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth', map_location=torch.device('cpu')))
    DCE_net.eval()

    # Define a dummy input tensor (match input size of your model)
    dummy_input = torch.randn(1, 3, 2000, 1500)  # Assuming input shape (1, C, H, W)

    # Export the model to ONNX
    onnx_filename = "enhance_model.onnx"
    torch.onnx.export(DCE_net, dummy_input, onnx_filename, 
                      export_params=True,  # Store trained parameters
                      opset_version=11,    # Use opset 11 for compatibility
                      do_constant_folding=True,  # Optimize constants
                      input_names=['input'], 
                      output_names=['output'])
    
    print(f"Model successfully exported to {onnx_filename}")

if __name__ == "__main__":
    export_to_onnx()

