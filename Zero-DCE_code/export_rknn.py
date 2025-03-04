from rknn.api import RKNN

ONNX_MODEL = 'enhance_model.onnx'  # Your exported ONNX model
RKNN_MODEL = 'enhance_model.rknn'  # Output RKNN model

def convert_model():
    rknn = RKNN()

    # Step 1: Configure RKNN
    print('--> Configuring RKNN')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]], target_platform='rk3588')
    print('RKNN configured successfully')

    # Step 2: Load ONNX model
    print('--> Loading ONNX model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Failed to load ONNX model')
        return
    print('ONNX model loaded successfully')

    # Step 3: Build RKNN model
    print('--> Building RKNN model')
    ret = rknn.build(do_quantization=False)  # Set to True for quantization
    if ret != 0:
        print('Failed to build RKNN model')
        return
    print('RKNN model built successfully')

    #    # Step 4: Debug model inputs/outputs **AFTER building**
    #    print("--> Checking model input/output shapes")
    #    input_details = rknn.get_input_tensor(0)
    #    output_details = rknn.get_output_tensor(0)

    #print(f"RKNN Input shape: {input_details['shape']}")
    #print(f"RKNN Output shape: {output_details['shape']}")

    # Step 5: Export RKNN model
    print('--> Exporting RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Failed to export RKNN model')
        return
    print(f'RKNN model saved as {RKNN_MODEL}')

if __name__ == '__main__':
    convert_model()

