import numpy

import chainer
from chainer import configuration
from chainer import cuda
from chainer.functions.normalization import batch_normalization
from chainer import initializers
from chainer import link
from chainer.utils import argument
from chainer import variable
from chainer.links import EmbedID
import chainer.functions as F
import chainer.links as L
from source.links.conditional_batch_normalization import ConditionalBatchNormalization


class SharedEmbeddingBatchNormalization(ConditionalBatchNormalization):
    """
    Conditional Batch Normalization
    Args:
        size (int or tuple of ints): Size (or shape) of channel
            dimensions.
        n_cat (int): the number of categories of categorical variable.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability.
        dtype (numpy.dtype): Type to use in computing.
        use_gamma (bool): If ``True``, use scaling parameter. Otherwise, use
            unit(1) which makes no effect.
        use_beta (bool): If ``True``, use shifting parameter. Otherwise, use
            unit(0) which makes no effect.
    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_
    .. seealso::
       :func:`~chainer.functions.batch_normalization`,
       :func:`~chainer.functions.fixed_batch_normalization`
    Attributes:
        gamma (~chainer.Variable): Scaling parameter.
        beta (~chainer.Variable): Shifting parameter.
        avg_mean (numpy.ndarray or cupy.ndarray): Population mean.
        avg_var (numpy.ndarray or cupy.ndarray): Population variance.
        N (int): Count of batches given for fine-tuning.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability. This value is added
            to the batch variances.
    """

    def __init__(self, in_size, out_size, decay=0.9, eps=2e-5, dtype=numpy.float32):
        super(SharedEmbeddingBatchNormalization, self).__init__(
            size=out_size, decay=decay, eps=eps, dtype=dtype)

        with self.init_scope():
            self.linear_gamma = L.Linear(in_size, out_size, nobias=True)
            self.linear_beta = L.Linear(in_size, out_size, nobias=True)

    def __call__(self, x, z, c, finetune=False, **kwargs):
        """__call__(self, x, c, finetune=False)
        Invokes the forward propagation of BatchNormalization.
        In training mode, the BatchNormalization computes moving averages of
        mean and variance for evaluatino during training, and normalizes the
        input using batch statistics.
        .. warning::
           ``test`` argument is not supported anymore since v2.
           Instead, use ``chainer.using_config('train', train)``.
           See :func:`chainer.using_config`.
        Args:
            x (Variable): Input variable.
            c (Variable): Input variable for conditioning gamma and beta
            finetune (bool): If it is in the training mode and ``finetune`` is
                ``True``, BatchNormalization runs in fine-tuning mode; it
                accumulates the input array to compute population statistics
                for normalization, and normalizes the input using batch
                statistics.
        """
        gamma_c = self.linear_gamma(F.concat([z, c])) + 1
        beta_c = self.linear_beta(F.concat([z, c]))
        return super(SharedEmbeddingBatchNormalization, self).__call__(x, gamma_c, beta_c, **kwargs)

    def start_finetuning(self):
        """Resets the population count for collecting population statistics.
        This method can be skipped if it is the first time to use the
        fine-tuning mode. Otherwise, this method should be called before
        starting the fine-tuning mode again.
        """
        self.N = 0
