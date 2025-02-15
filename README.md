
# Medical Image Classification with Deep Learning

Bu proje, tek bir kanser türünü (örneğin Kolon Kanseri) tespit etmek için bir derin öğrenme modeli geliştirmeyi amaçlar. Model, yüklenen tıbbi görüntüden kanser olup olmadığına dair bir olasılık değeri üretir (örneğin `%80 kanser` gibi).

## Proje Dizini

Aşağıda, **örnek** bir proje dizini gösterilmektedir (senin ekran görüntünde benzer bir yapı olduğu varsayılmıştır):

```
.
├── .cursor/
├── .venv/
├── data/
│   ├── cancer/
│   └── normal/
├── test/
├── train/
├── validation/
├── models/
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── test.py
│   └── train.py
├── ui/
├── .gitignore
├── README.md
└── training_loss.png
```

- `data/`: Eğitim verilerinin (örnek olarak `cancer` ve `normal` klasörleri) saklandığı ana klasör.
- `test/`, `train/`, `validation/`: İsteğe bağlı olarak ek test ve doğrulama verileri için alt klasörler.
- `models/`: Eğitilmiş model (`.pth`) dosyalarının saklanacağı klasör.
- `notebooks/`: Opsiyonel olarak Jupyter Notebook çalışmalarının konulabileceği klasör.
- `src/`: Proje ana kodları (veri yükleme, model tanımı, eğitim/test betikleri vb.).
- `ui/`: Qt veya başka bir arayüz dosyalarının tutulacağı klasör (opsiyonel).
- `.gitignore`: Git sürüm kontrolü için gereksiz dosyaları dışlayan ayar dosyası.
- `README.md`: Projeye genel bakış dosyası (bu dosya).
- `training_loss.png`: Eğitim sürecinde oluşan kayıp (loss) grafiğini gösteren resim.

---

## Kurulum ve Kullanım

1. **Proje Yapısını Oluşturma**

Eğer proje klasörlerini henüz otomatik oluşturmak istersen, `create_project_structure.py` (örnek bir betik) kullanabilirsin:

```bash
python create_project_structure.py
```

Bu, `.gitignore` dahil olmak üzere temel klasörleri oluşturur.

2. **Gerekli Kütüphanelerin Kurulumu**

Aşağıdaki komutlar ile PyTorch ve diğer kütüphaneleri kurabilirsin (Python 3.8+ önerilir):

```bash
pip install torch torchvision
pip install tqdm matplotlib
```

3. **Veri Hazırlığı**

- `data` klasörü altına `cancer` ve `normal` klasörlerini oluştur (örneğin `data/cancer`, `data/normal`).
- Kanserli örneklerin resimlerini `cancer`, sağlıklı örnekleri `normal` klasörüne koy.
- Opsiyonel olarak `test`, `train`, `validation` gibi ekstra klasörleri de kullanabilirsin.
- PyTorch ile eğitim yaparken, `train.py` betiği varsayılan olarak `data` klasöründen verileri okur (örnek olarak `%80` eğitim, `%20` validasyon şeklinde ayırır).

4. **Eğitim (train.py)**

- **--data_dir**: Eğitim verilerinin dizini (içinde `cancer` ve `normal` klasörleri bulunmalı).
- **--model_path**: Eğitilmiş modelin kaydedileceği dosya yolu (varsayılan: `models/cancer_detection_model.pth`).
- **--epochs**: Modeli kaç epoch boyunca eğiteceğin.
- **--batch_size**: Kaç örneğin aynı anda işleneceği.
- **--learning_rate**: Öğrenme oranı.

Eğitim tamamlandığında `cancer_detection_model.pth` dosyası `models/` klasörüne kaydedilir. Ayrıca `training_loss.png` dosyası oluşur.

5. **Tek Görsel Testi (test.py veya test_single.py)**

## Uyarılar

- **Tıbbi Veri Gizliliği**: Gerçek hasta verileri kullanıyorsan, yasal düzenlemelere (KVKK, HIPAA vb.) uymayı ve veri anonimleştirmesini yapmayı unutma.
- **Modelin Klinik Kullanımı**: Bu proje, yalnızca araştırma veya örnek amaçlıdır. Klinik kullanım için ek validasyon ve yasal onaylar gerekir.
- **GPU Kullanımı**: Eğer NVIDIA GPU varsa PyTorch otomatik olarak `cuda` desteğini kullanır, aksi halde `cpu` ile eğitim yapılır.

---

## Katkıda Bulunma

- Sorun bildirmek veya yeni özellik eklemek için **issue** açabilirsin.
- **Pull Request** göndererek projeye katkıda bulunabilirsin.
- Kod incelemeleri, dokümantasyon güncellemeleri ve hata düzeltmeleri değerlidir.

---

## Lisans

Bu proje [MIT Lisansı](https://opensource.org/licenses/MIT) altında sunulabilir. Dilersen lisansı değiştirebilir ya da kendi lisansını ekleyebilirsin.


**Teşekkürler!**  

Herhangi bir sorunda veya ek bir istekte bulunmak istersen bize ulaşabilirsin.

