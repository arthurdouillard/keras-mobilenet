# MobileNet

Implementation in Keras of MobileNet (v1).

[Arxiv link.](https://arxiv.org/abs/1704.04861)

MobileNet is a CNN network supposed to be efficient enough to work on mobile,
thus the name. Its efficiency comes from replacing convolution blocks by
depthwise separable convolution block: A depthwise conv followed by a pointwise conv.

# Depthwise

The depthwise conv has a filter per input channel. There are no interactions
between channels.

The number of input channels is thus the same as the number of output channels.

# Pointwise

The pointwise conv has, like usual conv, a filter per output channel and therefore
creates interaction between the channels. However its kernel size is 1x1.
