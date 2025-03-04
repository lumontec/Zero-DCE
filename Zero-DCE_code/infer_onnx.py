import onnxruntime as ort
import numpy as np
import cv2

# Load ONNX model
onnx_model = "./enhance_model.onnx"

# Create an inference session with explicit providers
ort_session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])

print("ONNX model loaded successfully!")

# Load and preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (1500, 2000))  # Resize to match model input
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    img = np.transpose(img, (2, 0, 1))  # Change shape to (C, H, W)
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, C, H, W)
    return img

# Path to the input image
image_path = "./data/test_data/PERS/low_road.jpg"  # Change this to your actual image path

# Preprocess input image
input_data = preprocess_image(image_path)

# Run inference
outputs = ort_session.run(None, {"input": input_data})

# Print output shape
output_image = outputs[0]
print(f"ONNX Output shape: {output_image.shape}")

# Remove batch dimension
output_image = output_image[0]  # Expected shape: (C, 1080, 1920)

# Check for extra channels
for i in range(output_image.shape[0]):
    print(f"ONNX Output Channel {i} mean: {output_image[i].mean()}")

if output_image.shape[0] > 3:
    print(f"⚠️ Warning: Model outputs {output_image.shape[0]} channels! Might be extra images.")

# Convert to (H, W, C) format for saving
output_image = np.transpose(output_image[:3], (1, 2, 0))  # Keep only the first 3 channels
output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)

# Save the output for inspection
output_path = "onnx_output.jpg"
cv2.imwrite(output_path, output_image)
print(f"Saved ONNX output image as '{output_path}'")

