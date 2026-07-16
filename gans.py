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
IMG_SHAPE = (28, 28, 1) # giris goruntusu boyutu

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

# Modeli oluşturalım
# generator = make_generator_model()

# discriminator modeli tanimla: gercek ve fake goruntuler arasinda ayirim yapacak
def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=IMG_SHAPE),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Flatten(), # 3D'yi duzlestir
        layers.Dense(1) # binary classification real/fake
    ])
    
    return model

# Modeli oluşturalım
# discriminator = make_discriminator_model()

# kayip (loss) function tanimla, optimizasyon algoritmasi tanimla
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output) # gercek = 1 etiketine sahip olsun
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output) # sahte goruntuler icin hedef 0 olsun
    return real_loss + fake_loss # toplam discriminator kaybi

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output) # generator sahte goruntuyu 1 gibi gostericek

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# yardimci fonksiyonlar tanimla: gercek ve fake goruntuler uret, discriminator ve generator modellerini guncelle
seed = tf.random.normal([16, NOISE_DIM]) # sabit gurultu ornegi

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False) # modeli sadece degerlendirme modunda calistir
    fig = plt.figure(figsize=(4, 4))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i, :, :, 0] + 1) / 2, cmap="gray") # goruntuleri 0 and 1 arasina rescale et
        plt.axis("off")
        
    if not os.path.exists("generated_images"):
        os.makedirs("generated_images")
        
    plt.savefig(f"generated_images/image_at_epoch_{epoch:04d}.png")
    plt.close()

# egitim fonksiyonu tanimla: generator ve discriminator modellerini egitecek