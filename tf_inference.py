import onnx
from onnx_tf.backend import prepare
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import utils
import dataset

model_onnx = onnx.load('./text_recognition.onnx')

tf_rep = prepare(model_onnx)

#Running Inference on an Image
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
converter = utils.strLabelConverter(alphabet)
transformer = dataset.resizeNormalize((100, 32))

img = Image.open('./data/res_lko.jpg')
img = img.convert('L')
img = transformer(img)
img = img.view(1, *img.size())
img = img.numpy()

output = tf_rep.run(img)[0]
output = torch.from_numpy(output)

_, output = output.max(2)
output = output.transpose(1, 0).contiguous().view(-1)

output_size = Variable(torch.IntTensor([output.size(0)]))
sim_pred = converter.decode(output.data, output_size.data, raw=False)
print()
print("-"*50)
print(sim_pred)