# Sparse-transformer
This is a implementation of sparse attention transformer deep learning model for langauge modeling. In sparse transformer the vanilla attention module is replaced by local windowed attention model. Local attention model has the advantages of reducing the quadratic complexity of vanilla attention model to nearly linear. This reduces the FLOPs to the computations from order (N^2 * d) to order (N * sqrt(N) * d). For example, for the same problem solved in my repository (https://github.com/maelhawary/Transformer-for-machine-translation.git) the vanilla transformer takes 800 seconds to solve 500 iteration, while the sparse local transformer takes 170 seconds on GPU RTXA6000. This allows training larger sequences in less computations time and therefore increase the model accuracy.

## Dataset
The model is learning to understand and generate a shakespeare note. The note file is named ('input.txt')

## Dependency
Python 3.8
Pytorch 2.1.0

## Train
$ python initial.py
