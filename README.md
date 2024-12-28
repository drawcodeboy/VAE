# VAE Implementation with PyTorch
### Exception
* Encoder: I didn't use flatten for extract latent vector, but 1 convolution layer + GAP
* Log Variance: Variance should be positive, so encoder infers log variance, which can take either positive or negative values.
* Decoder: Decoder takes a 16-dimensional input vector. I applied a linear transform to map it from 16 dims to 784, and then reshaped the resulting vector into a 28x28 matrix.