import torch
from torchviz import make_dot
import model  # Import your model definition

# Load the model
DCE_net = model.enhance_net_nopool()

# Create a dummy input tensor (assuming 1920x1080 image with 3 channels)
dummy_input = torch.randn(1, 3, 1920, 1080)

# Get the model output
output = DCE_net(dummy_input)

# Generate the computation graph
dot = make_dot(output, params=dict(DCE_net.named_parameters()))

# Save the visualization
dot.render("model_architecture", format="png")  # Saves as model_architecture.png

# Show the visualization (optional)
dot.view()

