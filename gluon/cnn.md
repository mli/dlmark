# Convolutional Neural Network

## Inference

### Throughput on various batch size

Given a batch size `b`, we feed `b` images, denoted by `X`, into `net` to measture the time `t` to run `net(X)`. Then the throughput is calculated by `b/t`.

```{.python .input}
import dlmark as dm
from bokeh.plotting import show, output_notebook
import json
import pandas as pd
output_notebook()
```

```{.python .input  n=1}
!cat cnn*throughput.json >thr.json
with open('thr.json', 'r') as f:
    thr = pd.DataFrame(json.load(f))
```

```{.python .input  n=2}
thr.model.unique()
```

```{.python .input  n=3}

p = dm.plot.batch_size_vs_throughput(thr[thr.model == 'VGG-11'])
show(p)
```

```{.python .input  n=15}
import numpy as np


models = ['DensetNet-121', 'ResNet-v1-101', 'VGG-11', 'AlexNet']

grid = dm.plot.batch_size_vs_throughput_grid(thr, models)

show(grid)
```

### Prediction accuracy versus throughput

```{.python .input  n=16}
!cat cnn*accuracy.json >acc.json
with open('acc.json', 'r') as f:
    acc = pd.DataFrame(json.load(f))

data = thr[(thr.model.isin(acc.model)) &
           (thr.batch_size.isin(acc.batch_size))]
data = data.set_index('model').join(acc[['model','accuracy']].set_index('model'))
data['model_prefix'] = [i[:i.rfind('-')] if i.rfind('-') > 0 else i for i in data.index]

```

```{.python .input}
p = dm.plot.throughput_vs_accuracy(data)
show(p)
```
