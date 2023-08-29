import torch

from nn_addition.add_model import AdditionModel

# load model
checkpoint_path = "tb_logs/addition_model/version_24/checkpoints/epoch=4-step=78125.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
# map_location only necessary if the model was trained on gpu and is now loaded on cpu
# that is not the case here, but it is useful to know

model = AdditionModel()
model.load_state_dict(checkpoint['state_dict']) # concept to load the weights into the model
model.eval()

# create dummy input
dummy_input = torch.tensor([[23.0, 42.0]])
# 2D tensor with shape [1, 2] (1 sample with 2 features) and data 23.0 and 42.0 (that should be added)

# convert model to onnx
try:
    torch.onnx.export(model, dummy_input, "addition_model_final.onnx")
    # torch.onnx.export(model, dummy_input, "addition_model.onnx", do_constant_folding=False)
    print("Model successfully exported to addition_model.onnx")
except Exception as e:
    print(f"Error exporting the model: {e}")
