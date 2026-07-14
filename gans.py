"""
GAN (Generative Adversarial Network) fashion mnist veri seti ile moda urunu tasarimi yapalim

Fashion MNIST veri seti 10 class iceren 28x28 boyutunda gri tonlamali goruntulerden olusur.
Fashion veri seti icerisinde 10 farkli class bulunur:
- T-shirt/top, sneaker, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot

Plan program?
Pip libraries?
import libraries?

"""

# import libraries
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.datasets import fashion_mnist

# veri seti yukle


# generator modeli tanimla: fake goruntuler uretecek


# discriminator modeli tanimla: gercek ve fake goruntuler arasinda ayirim yapacak


# kayip loss function tanimla, optimizasyon algoritmasi tanimla


# yardimci fonksiyonlar tanimla: gercek ve fake goruntuler uret, discriminator ve generator modellerini guncelle


# egitim fonksiyonu tanimla: generator ve discriminator modellerini egitecek