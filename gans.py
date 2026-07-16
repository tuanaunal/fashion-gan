"""
GAN (Generative Adversarial Network) fashion mnist veri seti ile moda urunu tasarimi yapalim

Fashion MNIST veri seti 10 class iceren 28x28 boyutunda gri tonlamali goruntulerden olusur.
Fashion veri seti icerisinde 10 farkli class bulunur:
- T-shirt/top, sneaker, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot

"""

# import libraries
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.datasets import fashion_mnist

BUFFER_SIZE = 60000 # veri seti boyutu
BATCH_SIZE = 128 # batch boyutu
NOISE_DIM = 100 # generatora verilecek gurultu vektorunun boyutu

# veri seti yukle
(train_images, _), (_, _) = fashion_mnist.load_data() # sadece goruntuleri al, etiketleri kullanma
train_images = train_images.reshape(-1, 28, 28, 1).astype("float32") # sekillendir ve floata cevir
train_images = (train_images - 127.5) / 127.5 # normalize et -1 ile 1 arasina
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE) # veri setini shuffle et ve batch'le

# generator modeli tanimla: fake goruntuler uretecek
def make_generator_model():
    model = tf.keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(NOISE_DIM,)), # ilk tam bagli katman, gurultuyu ozellik haritasina cevirir
        layers.BatchNormalization(), # egitim stabilitesini arttirir
        layers.LeakyReLU(), # negatif girisleri yumusatir
        
        layers.Reshape((7, 7, 256)), # tek boyutlu vektoru 3D ye donustur
        
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh")
    ])
    
    return model

# discriminator modeli tanimla: gercek ve fake goruntuler arasinda ayirim yapacak


# kayip loss function tanimla, optimizasyon algoritmasi tanimla


# yardimci fonksiyonlar tanimla: gercek ve fake goruntuler uret, discriminator ve generator modellerini guncelle


# egitim fonksiyonu tanimla: generator ve discriminator modellerini egitecek