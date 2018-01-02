# Convolutional Neural Network

We benchmark the convolutional neural networks provided by the [Gluon modelzoo](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html).

## Inference

### Throughput on various batch size

Given network `net` and batch size `b`, we feed `b` images, denoted by `X`, into `net` to measture the time `t` to complete `net(X)`. We then calculate the throughput as `b/t`. We first load the benchmark resutls and print all network and devices names

```{.python .input  n=1}
import dlmark as dm

thr = dm.benchmark.load_results('cnn.py__benchmark_throughput*json')

models = thr.model.unique()
devices = thr.device.unique()
(models, devices)
```

Now we visualize the throughput for each network when increasing the batch sizes. We only use the results on the first device and show a quater of networks:

```{.python .input  n=2}
from dlmark import plot
from bokeh.plotting import show, output_notebook
output_notebook()

data = thr[thr.device==devices[0]]
show(plot.batch_size_vs_throughput_grid(data, models[::4]))
```

The throughput increases with the batch size in log scale. The device memory, as exepcted, also increases linearly with the batch size. But note that, due to the pooled memory mechanism in MXNet, the measured device memory usage might be different to the actual memory usdage.

One way to measure the actual device memory usage is finding the largest batch size we can run.

```{.python .input  n=3}
bs = dm.benchmark.load_results('cnn.py__benchmark_largest_batch_size.json')    
show(plot.max_batch_size(bs))
```

## Throughput on various hardware

```{.python .input  n=4}
show(plot.throughput_vs_device(thr[(thr.model=='AlexNet')]))
```

```{.python .input  n=5}
show(plot.throughput_vs_device(thr[(thr.model=='ResNet-v2-50')]))
```

### Prediction accuracy versus throughput

We measture the prediction accuracy of each model using the ILSVRC 2012 validation dataset. Then plot the results together with the throughput with fixed batch size 64. We colorize models from the same family with the same color.

```{.python .input  n=6}
acc = dm.benchmark.load_results('cnn*accuracy.json')

data = thr[(thr.model.isin(acc.model)) &
           (thr.batch_size.isin(acc.batch_size)) &
           (thr.device.isin(acc.device))]
data = data.set_index('model').join(acc[['model','accuracy']].set_index('model'))
data['model_prefix'] = [i[:i.rfind('-')] if i.rfind('-') > 0 else i for i in data.index]

show(plot.throughput_vs_accuracy(data))
```
