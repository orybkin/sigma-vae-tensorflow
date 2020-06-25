# Simple and Effective VAE training with σ-VAE in TensorFlow

[[Project page]](https://orybkin.github.io/sigma-vae/) [[Colab]](https://colab.research.google.com/drive/1XstWM57-LyIogBCcKgtvUmo6b7bVhuNM?usp=sharing) [[PyTorch implementation]](https://github.com/orybkin/sigma-vae-pytorch) 

This is the TensorFlow implementation of the σ-VAE paper. See the σ-VAE project page for more info, results, and alternative
 implementations. Also see the Colab version of this repo to train a sigma-VAE with zero setup needed!

This implementation is based on [this](https://github.com/LynnHo/VAE-Tensorflow) VAE implementation. In contrast to the original implementation,  the σ-VAE 
achieves good results without tuning the heuristic weight beta since the decoder variance balances the objective. 
It is also very easy to implement, check out individual commits to see the few lines of code you need to add this to your VAE.!

## How to run it 

This repo implements several VAE versions.

First, a VAE from the original implementation from github that uses MSE loss. This implementation works very poorly because
the MSE loss averages the pixels instead of summing them. Don't do this! You have to sum the loss across pixels and
latent dimensions according to the definition of multivariate Gaussian (and other) distributions.
```
python train.py --experiment_name mse --model mse
```

Summing the loss works a bit better and is equivalent to the Gaussian negative log likelihood (NLL) with a certain, constant 
variance. This second model uses the Gaussian NLL as the reconstruction term. However, since the variance is constant
it is still unable to balance the reconstruction and KL divergence term.
```
python train.py --experiment_name gaussian --model gaussian
```

The third model is the σ-VAE. It learns the variance of the decoding distribution, which works significantly better and produces
high-quality samples. This is because learning the variance automatically balances the VAE objective. One could balance 
the objective manually by using beta-VAE, however, this is not required when learning the variance!
```
python train.py --experiment_name sigma --model sigma
```

Finally, optimal sigma-VAE uses a batch-wise analytic estimate of the variance, which speeds up learning and improves results.
It is also extremely easy to implement! 
```
python train.py --experiment_name optimal --model optimal
```