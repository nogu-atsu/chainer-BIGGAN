import chainer
from chainer import functions as F
from chainer import links as L
from source.links.shared_embedding_batch_normalization import SharedEmbeddingBatchNormalization

import numpy as np


def _upsample(x):
    h, w = x.shape[2:]
    return F.unpooling_2d(x, 2, outsize=(h * 2, w * 2))


def upsample_conv(x, conv):
    return conv(_upsample(x))

def calc_spectral_norm(w, u):
    w = w.reshape(-1, u.shape[1])
    v_ = np.dot(u, w.T)
    v_ = v_ / np.sqrt((np.sum(np.square(v_))+1e-4))
    u_ = np.dot(v_, w)
    u_ = u_ / np.sqrt((np.sum(np.square(u_))+1e-4))
    return np.dot(np.dot(u_, w.T), v_.T)

def copy_conv(link, w, b=None, u=None):
    if u is not None:
        w = w / calc_spectral_norm(w, u)
    link.W.data[:] = w.transpose(3, 2, 0, 1)
    if b is not None:
        link.b.data[:] = b


def copy_hbn(link, gamma, gamma_u, beta, beta_u, avg_mean, avg_var):
    copy_linear(link.linear_gamma, gamma, u=gamma_u)
    copy_linear(link.linear_beta, beta, u=beta_u)
    link.avg_mean[:] = avg_mean
    link.avg_var[:] = avg_var
    # layer.eps = eps


def copy_bn(link, gamma, beta, avg_mean, avg_var):
    link.beta.data[:] = beta
    link.gamma.data[:] = gamma
    link.avg_mean[:] = avg_mean
    link.avg_var[:] = avg_var


def copy_linear(link, w, b=None, u=None):
    if u is not None:
        w = w / calc_spectral_norm(w, u)
    link.W.data[:] = w.transpose()
    if b is not None:
        link.b.data[:] = b


class ResBlock(chainer.Chain):
    def __init__(self, in_size, in_channel, out_channel):
        layers = {}
        layers["bn1"] = SharedEmbeddingBatchNormalization(in_size, in_channel)
        layers["conv1"] = L.Convolution2D(in_channel, out_channel, 3, 1, 1)
        layers["bn2"] = SharedEmbeddingBatchNormalization(in_size, out_channel)
        layers["conv2"] = L.Convolution2D(out_channel, out_channel, 3, 1, 1)
        layers["conv_sc"] = L.Convolution2D(in_channel, out_channel, 1, 1, 0)
        super(ResBlock, self).__init__(**layers)

    def __call__(self, x, z, c):
        h = self.bn1(x, z, c)
        h = upsample_conv(F.relu(h), self.conv1)
        h = self.conv2(F.relu(self.bn2(h, z, c)))
        return h + upsample_conv(x, self.conv_sc)

    def copy_from_tf(self, weights_dict, name):
        copy_conv(self.conv1, weights_dict[name + '/conv0/w/ema_b999900'],
                  weights_dict[name + '/conv0/b/ema_b999900'],
                  weights_dict[name + '/conv0/u0'], )
        copy_conv(self.conv2, weights_dict[name + '/conv1/w/ema_b999900'],
                  weights_dict[name + '/conv1/b/ema_b999900'],
                  weights_dict[name + '/conv1/u0'])
        copy_conv(self.conv_sc, weights_dict[name + '/conv_sc/w/ema_b999900'],
                  weights_dict[name + '/conv_sc/b/ema_b999900'],
                  weights_dict[name + '/conv_sc/u0'])
        copy_hbn(self.bn1, weights_dict[name + '/HyperBN/gamma/w/ema_b999900'],
                 weights_dict[name + '/HyperBN/gamma/u0'],
                 weights_dict[name + '/HyperBN/beta/w/ema_b999900'],
                 weights_dict[name + '/HyperBN/beta/u0'],
                 weights_dict[name + '/CrossReplicaBN/accumulated_mean'],
                 weights_dict[name + '/CrossReplicaBN/accumulated_var'])
        copy_hbn(self.bn2, weights_dict[name + '/HyperBN_1/gamma/w/ema_b999900'],
                 weights_dict[name + '/HyperBN_1/gamma/u0'],
                 weights_dict[name + '/HyperBN_1/beta/w/ema_b999900'],
                 weights_dict[name + '/HyperBN_1/beta/u0'],
                 weights_dict[name + '/CrossReplicaBN_1/accumulated_mean'],
                 weights_dict[name + '/CrossReplicaBN_1/accumulated_var'])


class NonLocalBlock(chainer.Chain):
    def __init__(self, ch):
        self.ch = ch
        layers = {}
        layers["f_conv"] = L.Convolution2D(ch, ch // 8, 1, 1, 0, nobias=True)
        layers["g_conv"] = L.Convolution2D(ch, ch // 8, 1, 1, 0, nobias=True)
        layers["h_conv"] = L.Convolution2D(ch, ch // 2, 1, 1, 0, nobias=True)
        layers["o_conv"] = L.Convolution2D(ch // 2, ch, 1, 1, 0, nobias=True)
        layers["gamma"] = L.Parameter(np.array(0, dtype="float32"))
        super(NonLocalBlock, self).__init__(**layers)

    def __call__(self, x):
        batchsize, _, w, _ = x.shape
        f = self.f_conv(x).reshape(batchsize, self.ch // 8, -1)
        g = self.g_conv(x)
        g = F.max_pooling_2d(g, 2, 2).reshape(batchsize, self.ch // 8, -1)
        attention = F.softmax(F.matmul(f, g, transa=True), axis=2)
        h = self.h_conv(x)
        h = F.max_pooling_2d(h, 2, 2).reshape(batchsize, self.ch // 2, -1)
        o = F.matmul(h, attention, transb=True).reshape(batchsize, self.ch // 2, w, w)
        o = self.o_conv(o)
        return x + self.gamma.W * o

    def copy_from_tf(self, weights_dict, name):
        copy_conv(self.f_conv, weights_dict[name + '/theta/w/ema_b999900'],
                  u=weights_dict[name + '/theta/u0'])
        copy_conv(self.g_conv, weights_dict[name + '/phi/w/ema_b999900'],
                  u=weights_dict[name + '/phi/u0'])
        copy_conv(self.h_conv, weights_dict[name + '/g/w/ema_b999900'],
                  u=weights_dict[name + '/g/u0'])
        copy_conv(self.o_conv, weights_dict[name + '/o_conv/w/ema_b999900'],
                  u=weights_dict[name + '/o_conv/u0'])
        self.gamma.W.data = weights_dict[name + '/gamma/ema_b999900']


class Generator(chainer.Chain):
    def __init__(self, ch=96):
        self.ch = ch
        layers = {}
        layers["linear"] = L.Linear(1000, 128, nobias=True)
        layers["dense"] = L.Linear(20, 4 * 4 * 16 * ch)
        layers["resblock1"] = ResBlock(148, 16 * ch, 16 * ch)
        layers["resblock2"] = ResBlock(148, 16 * ch, 8 * ch)
        layers["resblock3"] = ResBlock(148, 8 * ch, 8 * ch)
        layers["resblock4"] = ResBlock(148, 8 * ch, 4 * ch)
        layers["resblock5"] = ResBlock(148, 4 * ch, 2 * ch)
        layers["non_local6"] = NonLocalBlock(2 * ch)
        layers["resblock7"] = ResBlock(148, 2 * ch, ch)
        layers["bn"] = L.BatchNormalization(ch)
        layers["conv"] = L.Convolution2D(ch, 3, 3, 1, 1)
        super(Generator, self).__init__(**layers)

    def forward(self, z, c, layers=[]):
        output = {}
        c = self.linear(c)
        if "c_linear" in layers:
            output["c_linear"] = c
        z = F.split_axis(z, 7, axis=1)
        if "z_split" in layers:
            output["z_split"] = z
        h = self.dense(z[0]).reshape(-1, 4, 4, 16 * self.ch).transpose(0, 3, 1, 2)
        if "z_linear" in layers:
            output["z_linear"] = h
        h = self.resblock1(h, z[1], c)
        if "res1" in layers:
            output["res1"] = h
        h = self.resblock2(h, z[2], c)
        if "res2" in layers:
            output["res2"] = h
        h = self.resblock3(h, z[3], c)
        if "res3" in layers:
            output["res3"] = h
        h = self.resblock4(h, z[4], c)
        if "res4" in layers:
            output["res4"] = h
        h = self.resblock5(h, z[5], c)
        if "res5" in layers:
            output["res5"] = h
        h = self.non_local6(h)
        if "non6" in layers:
            output["non6"] = h
        h = self.resblock7(h, z[6], c)
        if "res7" in layers:
            output["res7"] = h
        h = F.relu(self.bn(h))
        h = F.tanh(self.conv(h))
        if output:
            return h, output
        return h

    def copy_params_from_tf(self, weights_dict):
        copy_linear(self.linear, weights_dict['linear/w/ema_b999900'])
        copy_linear(self.dense, weights_dict['Generator/G_Z/G_linear/w/ema_b999900'],
                    weights_dict['Generator/G_Z/G_linear/b/ema_b999900'],
                    weights_dict['Generator/G_Z/G_linear/u0'])
        self.resblock1.copy_from_tf(weights_dict, 'Generator/GBlock')
        self.resblock2.copy_from_tf(weights_dict, 'Generator/GBlock_1')
        self.resblock3.copy_from_tf(weights_dict, 'Generator/GBlock_2')
        self.resblock4.copy_from_tf(weights_dict, 'Generator/GBlock_3')
        self.resblock5.copy_from_tf(weights_dict, 'Generator/GBlock_4')
        self.non_local6.copy_from_tf(weights_dict, 'Generator/attention')
        self.resblock7.copy_from_tf(weights_dict, 'Generator/GBlock_5')
        copy_bn(self.bn, weights_dict['Generator/ScaledCrossReplicaBN/gamma/ema_b999900'],
                weights_dict['Generator/ScaledCrossReplicaBN/beta/ema_b999900'],
                weights_dict['Generator/ScaledCrossReplicaBNbn/accumulated_mean'],
                weights_dict['Generator/ScaledCrossReplicaBNbn/accumulated_var'])
        copy_conv(self.conv, weights_dict['Generator/conv_2d/w/ema_b999900'],
                  weights_dict['Generator/conv_2d/b/ema_b999900'],
                  weights_dict['Generator/conv_2d/u0'])
