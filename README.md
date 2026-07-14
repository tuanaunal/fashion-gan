# GAN ile Moda Ürün Tasarımı (Fashion-GAN)

Bu proje, **Generative Adversarial Networks (Çekişmeli Üretici Ağlar - GAN)** mimarisini kullanarak sıfırdan yapay zekayla yeni moda ürünleri tasarlamayı amaçlamaktadır. Projede popüler **Fashion-MNIST** veri seti kullanılarak tişört, ayakkabı, çanta ve bot gibi kıyafetlerin yapay zeka tarafından sıfırdan üretilmesi hedeflenmiştir.

---

## Proje Planı

Proje, her aşaması modüler ve adım adım commit edilerek GitHub üzerinde geliştirilecektir:

- **Adım 1:** Proje iskeletinin oluşturulması ve kütüphanelerin import edilmesi (Tamamlandı)
- **Adım 2:** Veri kümesinin yüklenmesi ve ön işleme adımları ([-1, 1] normalizasyonu)
- **Adım 3:** **Generator** (Üretici) modelinin tasarımı
- **Adım 4:** **Discriminator** (Ayırt Edici) modelinin tasarımı
- **Adım 5:** Loss fonksiyonları ve optimizasyon algoritmalarının tanımlanması
- **Adım 6:** Eğitim döngüsünün (Training Loop) kurulması
- **Adım 7:** Sonuçların görselleştirilmesi ve üretilen tasarımların kaydedilmesi

---

## Kullanılan Teknolojiler ve Kütüphaneler

- **Python 3.12**
- **TensorFlow / Keras** (Derin öğrenme modellerinin kurulması ve eğitilmesi için)
- **NumPy** (Matris operasyonları ve veri ön işleme için)
- **Matplotlib** (Üretilen moda tasarımlarını görselleştirmek için)

---

## Dosya Yapısı

```text
fashion-gan/
│
├── venv/                  # Sanal Ortam (Virtual Environment)
├── gans.py                # Ana proje scripti (Model tanımları ve eğitim döngüsü)
├── requirements.txt       # Gerekli kütüphaneler listesi
└── README.md              # Proje dökümantasyonu (Şu an buradasınız)
```
