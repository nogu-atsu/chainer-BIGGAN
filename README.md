# Chainer implementation of BIGGAN (in progress)

BIGGAN [[paper]](https://arxiv.org/abs/1809.11096)


Chainer implementation of BIGGAN only for inference. You can reproduce the results of the official tensorflow hub module ([tf hub](https://tfhub.dev/deepmind/biggan-256/2)).
## Requirements
- python3.6
- tensorflow=1.12
- tensorflow-hub=0.2.0
- chainer>=5.0.0
- cupy>=5.0.0

## All you have to do are...
### 1. Run copy_weights.ipynb and save weights
You can use colaboratory if going wrong.
### 2. Run biggan.ipynb

That's all

## Generative results
![10](https://github.com/nogu-atsu/chainer-BIGGAN/blob/master/figs/10.png "10")
![232](https://github.com/nogu-atsu/chainer-BIGGAN/blob/master/figs/232.png "232")
![933](https://github.com/nogu-atsu/chainer-BIGGAN/blob/master/figs/933.png "933")
