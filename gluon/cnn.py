import dlmark as dm
import mxnet as mx
from mxnet import nd
import time
import numpy as np
import json
from mxnet.gluon.model_zoo import vision as models

def _preprocess(X):
    rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
    rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
    X = nd.array(X).transpose((0,3,1,2))
    return (X.astype('float32') / 255 - rgb_mean) / rgb_std

#'MobileNet-width-0.25':models.mobilenet0_25,
#'MobileNet-width-0.5':models.mobilenet0_5,
#'MobileNet-width-0.75':models.mobilenet0_75,
#'MobileNet':models.mobilenet1_0,
#'ResNet-101-v2':models.resnet101_v2,
#'ResNet-152-v2':models.resnet152_v2,
# 'VGG-11-BN':models.vgg11_bn,
# 'VGG-13-BN':models.vgg13_bn,
# 'VGG-16-BN':models.vgg16_bn,
# 'VGG-19-BN':models.vgg19_bn

# modelzoo = {
#     ('AlexNet', ''):models.alexnet,
#     ('DensetNet','121'):models.densenet121,
#     ('DensetNet','161'):models.densenet161,
#     ('DensetNet','169'):models.densenet169,
#     ('DensetNet','201'):models.densenet201,
#     ('ResNet-v1','101'):models.resnet101_v1,
#     ('ResNet-v1','152'):models.resnet152_v1,
#     ('ResNet-v1','18'):models.resnet18_v1,
#     ('ResNet-v1','34'):models.resnet34_v1,
#     ('ResNet-v1','50'):models.resnet50_v1,
#     ('ResNet-v2','18'):models.resnet18_v2,
#     ('ResNet-v2','34'):models.resnet34_v2,
#     ('ResNet-v2','50'):models.resnet50_v2,
#     ('SqueezeNet','1.0'):models.squeezenet1_0,
#     ('SqueezeNet','1.1'):models.squeezenet1_1,
#     ('VGG','11'):models.vgg11,
#     ('VGG','13'):models.vgg13,
#     ('VGG','16'):models.vgg16,
#     ('VGG','19'):models.vgg19,
# }

modelzoo = {
    'AlexNet':models.alexnet,
    'DensetNet-121':models.densenet121,
    'DensetNet-161':models.densenet161,
    'DensetNet-169':models.densenet169,
    'DensetNet-201':models.densenet201,
    'ResNet-v1-101':models.resnet101_v1,
    'ResNet-v1-152':models.resnet152_v1,
    'ResNet-v1-18':models.resnet18_v1,
    'ResNet-v1-34':models.resnet34_v1,
    'ResNet-v1-50':models.resnet50_v1,
    'ResNet-v2-18':models.resnet18_v2,
    'ResNet-v2-34':models.resnet34_v2,
    'ResNet-v2-50':models.resnet50_v2,
    'SqueezeNet-1.0':models.squeezenet1_0,
    'SqueezeNet-1.1':models.squeezenet1_1,
    'VGG-11':models.vgg11,
    'VGG-13':models.vgg13,
    'VGG-16':models.vgg16,
    'VGG-19':models.vgg19,
}
def get_accuracy(model_name):
    batch_size = 64
    dataset = dm.image.ILSVRC12Val(batch_size, 'http://xx/', root='/home/ubuntu/imagenet_val/')
    ctx = mx.gpu(0)
    net = modelzoo[model_name](pretrained=True)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()

    n, acc = 0, 0
    for X, y in dataset:
        X = _preprocess(X).as_in_context(ctx)
        y = nd.array(y, ctx)
        yhat = net(X)
        acc += nd.sum(yhat.argmax(axis=1)==y).asscalar()
        n += X.shape[0]
        if n > 5e3:
            break
    return {
        'device':dm.utils.nv_gpu_name(0),
        'model':model_name,
        'batch_size':batch_size,
        'accuracy':acc/n,
        'workload':'Inference',
    }

def benchmark_accuracy():
    device_name = dm.utils.nv_gpu_name(0).replace(' ', '-').lower()
    results = []
    for model_name in modelzoo:
        print(model_name)
        res = dm.benchmark.run_with_separate_process(
            get_accuracy, model_name
        )
        results.append(res)
        with open('cnn_'+device_name+'_accuracy.json', 'w') as f:
            json.dump(results, f)

# benchmark_accuracy()

def get_throughput(model_name, batch_size):
    ctx = mx.gpu(0)
    device_name = dm.utils.nv_gpu_name(0)
    net = modelzoo[model_name](pretrained=True)
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    mem = dm.utils.nv_gpu_mem_usage()

    # warm up
    X = np.random.uniform(low=-254, high=254, size=(batch_size,224,224,3))
    X = _preprocess(X).as_in_context(ctx)
    net(X).wait_to_read()

    # iterate mutliple times
    iters = 1000 // batch_size
    tic = time.time()
    for _ in range(iters):
        net(X).wait_to_read()
    nd.waitall()
    throughput = iters*batch_size/(time.time()-tic)

    return {
        'device':device_name,
        'model':model_name,
        'batch_size':batch_size,
        'throughput':throughput,
        'workload':'Inference',
        'device_mem':dm.utils.nv_gpu_mem_usage() - mem
    }

def benchmark_throughput():
    results = []
    device_name = dm.utils.nv_gpu_name(0).replace(' ', '-').lower()
    for model_name in modelzoo:
        print(model_name)
        # batch_sizes = [1,2,4,8,16,32,64]
        batch_sizes = [1024, 10000]
        if not 'VGG' in model_name:
            batch_sizes += [128,]
        for batch_size in batch_sizes:
            res = dm.benchmark.run_with_separate_process(
                get_throughput, model_name, batch_size
            )
            results.append(res)

        with open('cnn_'+device_name+'_throughput.json', 'w') as f:
            json.dump(results, f)

benchmark_throughput()
