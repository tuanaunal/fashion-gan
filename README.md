# Fashion-GAN: Fashion-MNIST ile Sentetik Kıyafet Üretimi

Bu proje, **Fashion-MNIST** veri setini kullanarak sıfırdan yapay zeka ile modern kıyafet, ayakkabı ve çanta tasarımları üreten bir **Derin Yapay Çekişmeli Ağ (Deep Convolutional GAN - DCGAN)** modelidir. Projede Keras'ın hazır eğitim döngüleri yerine, derin öğrenme mimarilerine tam hakimiyet sağlamak amacıyla **`tf.GradientTape` ile tamamen özel bir eğitim döngüsü (Custom Training Loop)** inşa edilmiştir.

---

## Proje Mimarisi & Çalışma Mantığı

GAN (Generative Adversarial Networks) mimarisi, oyun teorisindeki minimax prensibine dayanarak birbiriyle çekişen iki farklı yapay sinir ağının ortaklaşa eğitilmesini hedefler:

1. **Jeneratör (Generator):** Standart normal dağılımdan alınan rastgele 100 boyutlu gürültü (`NOISE_DIM`) vektörlerini girdi olarak alır. `Conv2DTranspose` katmanlarını kullanarak bu karıncalı veriyi adım adım $28 \times 28$ boyutlarında gerçekçi kıyafet görsellerine dönüştürür.
2. **Diskriminatör / Dedektif (Discriminator):** Önüne gelen bir görselin veri setindeki gerçek bir moda ürünü mü yoksa jeneratörün ürettiği sahte bir tasarım mı olduğunu ayırt etmeye çalışan kararlı bir sınıflandırıcıdır.

---

## Teknik Özellikler ve Geliştirme Süreci

- **Özel Kayıp Fonksiyonları (Custom Loss):** İki modelin taban tabana zıbıt hedeflerini yönetebilmek adına `BinaryCrossentropy(from_logits=True)` tabanlı `generator_loss` ve `discriminator_loss` fonksiyonları özel olarak formüle edilmiştir.
- **Dinamik Eğitim Döngüsü (`tf.GradientTape`):** Modellerin ağırlık güncellemeleri, her veri paketinde (batch) gradyanların anlık takibi ve `Adam` optimizasyon algoritmasına (`1e-4` öğrenme oranı ile) beslenmesiyle tamamen elle kontrol edilmiştir.
- **Zaman Tüneli Takibi (Inference & Evaluation):** Eğitim süreci boyunca jeneratörün katettiği yolu objektif izleyebilmek adına sabit bir gürültü matrisi (`seed`) tanımlanmıştır. Her epoch sonunda bu sabit tohum üzerinden üretilen $4 \times 4$'lük 16 görsel otomatik olarak `generated_images/` klasörüne kaydedilmektedir.

---

## Proje Dosya Yapısı

```text
project_4/
│
├── gans.py             # Tüm GAN mimarisi, özel loss fonksiyonları ve eğitim döngüsü
├── README.md           # Proje dokümantasyonu
│
├── venv/               # Python Sanal Ortamı
└── generated_images/   # Eğitim sırasında epoch epoch kaydedilen sentetik tasarımlar
```
