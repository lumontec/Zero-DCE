import torch
import model  # Import your model definition

# Load the model
DCE_net = model.enhance_net_nopool()

# Print a summary of the model architecture
print(DCE_net)

# Alternatively, print a more detailed summary (if torchinfo is installed)
try:
    from torchinfo import summary
    print(summary(DCE_net, input_size=(1, 3, 1920, 1080)))  # Assuming input image size is 1920x1080
except ImportError:
    print("Install 'torchinfo' for a detailed summary: pip install torchinfo")

