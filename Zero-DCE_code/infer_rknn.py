import numpy as np
import cv2
from rknn.api import RKNN

RKNN_MODEL = 'enhance_model.rknn'  # Path to your RKNN model

def preprocess_image(image_path):
    """ Load and preprocess image for RKNN inference """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (1500, 2000))  # Resize to match model input
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1] range
    img = np.transpose(img, (2, 0, 1))  # Convert shape from (H, W, C) to (C, H, W)
    img = np.expand_dims(img, axis=0)  # Add batch dimension -> (1, C, H, W)
    return img

def run_inference(image_path):
    rknn = RKNN()

    # Load RKNN model
    print('--> Loading RKNN model')
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Failed to load RKNN model')
        return
    print('RKNN model loaded successfully')

    # Initialize RKNN runtime on RK3588
    print('--> Initializing RKNN runtime')
    ret = rknn.init_runtime(target='rk3588')
    if ret != 0:
        print('Failed to initialize RKNN runtime')
        return
    print('RKNN runtime initialized')

    # Preprocess input image
    input_data = preprocess_image(image_path)

    # Run inference
    print('--> Running inference')
    outputs = rknn.inference(inputs=[input_data], data_format='nchw')
    print('Inference complete')

    # Convert output to image format **before** any postprocessing
    raw_output = outputs[0]  # Assuming single output tensor
    print(f"Raw output shape: {raw_output.shape}")  # Debug shape

    raw_image = raw_output.squeeze(0)  # Remove batch dimension -> (C, H, W)
    raw_image = np.transpose(raw_image, (1, 2, 0))  # Convert to (H, W, C)
    
    # Save raw output to check before postprocessing
    debug_path = image_path.replace("test_data", "debug").replace(".jpg", "_raw_output.jpg")
    cv2.imwrite(debug_path, raw_image * 255)  # Scale back to [0,255]
    print(f"Raw model output saved at {debug_path}")

    # Release RKNN
    rknn.release()

if __name__ == '__main__':
    image_path = './data/test_data/PERS/low_road.jpg'  # Change this to your actual test image
    run_inference(image_path)

