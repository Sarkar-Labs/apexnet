here"""ApexNet: an explicit, high-level API covering (nearly) all of TensorFlow's building blocks."""
from . import (callbacks, conv, data, hardware, interop, layers, losses, models, optim, preprocessing,
               regularizers, rnn, selftest, tune, utils)
from .callbacks import (BackupAndRestore, Callback, Checkpoint, CSVLogger, EarlyStopping,
                        GradientHealth, LambdaCallback, LearningRateScheduler, Logger, Profiler,
                        ReduceLROnPlateau, TensorBoard, TerminateOnNaN)
from .conv import (AvgPool1D, AvgPool2D, AvgPool3D, Conv1D, Conv1DTranspose, Conv2D,
                   Conv2DTranspose, Conv3D, Conv3DTranspose, DepthwiseConv2D, GlobalAvgPool,
                   GlobalAvgPool1D, GlobalAvgPool2D, GlobalMaxPool, MaxPool1D, MaxPool2D,
                   MaxPool3D, PixelShuffle, SeparableConv2D, SqueezeExcite, Upsample, ZeroPad)
from .core import (Module, Sequential, from_config, load_model, register, save_model,
                   set_precision)
from .interop import Wrap, from_saved_model, to_keras
from .interop_jax import Jax
from .interop_torch import Torch
from .layers import (GLU, Activation, BatchNorm, Bilinear, Dense, DropPath, Dropout, Einsum,
                     Embedding, Flatten, GaussianNoise, GroupNorm, InstanceNorm, Lambda,
                     LayerNorm, MultiHeadAttention, Parallel, Permute, PositionalEncoding,
                     PReLU, Remat, Reshape, Residual, RMSNorm, SpatialDropout, SpectralNorm,
                     TransformerBlock)
from .losses import (AUC, F1, MAE, MSE, R2, Accuracy, BinaryAccuracy, MacroF1, MeanOf, Metric,
                     Perplexity, Precision, Recall, TopKAccuracy, binary_cross_entropy,
                     categorical_cross_entropy, cosine_distance, cross_entropy, ctc_loss,
                     dice_loss, distillation, focal_loss, hinge, huber, info_nce, kl_divergence,
                     log_cosh, mae, mse, poisson, squared_hinge, triplet_loss)
from .optim import (SGD, Adagrad, Adam, Adamax, AdamW, Lion, Lookahead, RMSprop, constant, cosine_restarts,
                    exponential_decay, inverse_time_decay, one_cycle, piecewise_constant,
                    polynomial_decay, warmup_cosine, warmup_linear, with_warmup)
from .preprocessing import (MelSpectrogram, Normalization, RandomBrightness, RandomContrast,
                            RandomCrop, RandomFlip, RandomRotation, RandomTranslation, Rescaling,
                            Resizing, Spectrogram, TextVectorizer)
from .rnn import (GRU, LSTM, RNN, Bidirectional, GRUCell, LSTMCell, SimpleRNN, SimpleRNNCell)
from .trainer import Trainer
from .hardware import auto
from .utils import auto_strategy, set_seed

tf = __import__("tensorflow")  # `apexnet.tf`: the full TensorFlow is always one attribute away
__version__ = "0.6.0"
