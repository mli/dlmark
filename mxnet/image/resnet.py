import dlmark
import mxnet as mx
import time
from mxnet.gluon.model_zoo import vision as models

net = models.resnet18_v2(pretrained=True, ctx=mx.gpu(0))
# net.collect_params().reset_ctx(mx.gpu(0))

batch_size = 2
x = mx.nd.random.uniform(shape=(batch_size, 3, 224, 224), ctx=mx.gpu(0))

net(x).wait_to_read()

tic = time.time()
for i in range(100):
    y = net(x)
    mx.nd.waitall()

print((time.time()-tic)/100*1000)
