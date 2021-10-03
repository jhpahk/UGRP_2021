from key_encoder_ver2 import Encoder

import torch
import onnx
import numpy as np
import tensorflow as tf
from onnx_tf.backend import prepare

model = Encoder()
model.eval()

example = torch.rand(1, 3, 256, 256)
torch.onnx.export(
    model,
    example,
    "encoder.onnx",
    opset_version=10
)

onnx_model = onnx.load("encoder.onnx")
onnx.checker.check_model(onnx_model)

tf_rep = prepare(onnx_model)
tf_rep.export_graph("encoder_tf")

converter = tf.lite.TFLiteConverter.from_saved_model("encoder_tf")
converter_quantized = tf.lite.TFLiteConverter.from_saved_model("encoder_tf")

# float16 quantization
converter_quantized.optimizations = [tf.lite.Optimize.DEFAULT]
converter_quantized.target_spec.supported_types = [tf.float16]

model_tflite = converter.convert()
with open("encoder.tflite", "wb") as f:
    f.write(model_tflite)

model_tflite_quantized = converter_quantized.convert()
with open("encoder_quantized.tflite", "wb") as f_q:
    f_q.write(model_tflite_quantized)
