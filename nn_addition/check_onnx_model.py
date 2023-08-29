import onnx
import onnxruntime as ort


def check():
    """ DEBUG: Check if the ONNX model is loaded properly """
    try:
        # load the ONNX model
        model_onnx = onnx.load("addition_model.onnx")
        # check the model
        onnx.checker.check_model(model_onnx)
        print("The model is checked!")
    except Exception as e:
        print(f"Error checking the model: {e}")


def check_runtime():
    ort_session = ort.InferenceSession("addition_model.onnx")

    inputs = {ort_session.get_inputs()[0].name: [[23.0, 42.0]]}
    outputs = ort_session.run(None, inputs)
    print(outputs)


if __name__ == "__main__":
    # check()
    check_runtime()
