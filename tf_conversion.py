import onnx
from onnx_tf.backend import prepare
import numpy as np

model_onnx = onnx.load('./text_recognition.onnx')

tf_rep = prepare(model_onnx)

# Print out tensors and placeholders in model (helpful during inference in TensorFlow)
print(tf_rep.tensor_dict)

# Export model as .pb file
tf_rep.export_graph('./model_simple.pb')
