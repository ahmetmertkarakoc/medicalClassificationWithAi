import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class MedicalImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # data_dir: Görsellerin bulunduğu ana dizin (içinde 'cancer' ve 'normal' klasörleri olmalı).
        # transform: Görsellere uygulanacak ön işleme (örnek: yeniden boyutlandırma, normalize etme).
        
        self.data_dir = data_dir
        
        # Eğer dönüşüm belirtilmediyse, varsayılan bir dönüşüm dizisi oluştur.
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),     # Görselleri 224x224 boyutuna getir.
            transforms.ToTensor(),            # Pikselleri PyTorch tensörüne dönüştür.
            transforms.Normalize(             # Renk kanallarını normalle.
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Görsellerin dosya yollarını ve etiketlerini tutacağımız listeler.
        self.images = []
        self.labels = []
        
        # 'cancer' ve 'normal' klasörlerinin varlığını kontrol et.
        cancer_dir = os.path.join(data_dir, 'cancer')
        normal_dir = os.path.join(data_dir, 'normal')
        
        if not os.path.exists(cancer_dir) or not os.path.exists(normal_dir):
            raise ValueError(
                f"Veri dizini yapısı hatalı. '{data_dir}' altında "
                "'cancer' ve 'normal' klasörleri bulunamadı."
            )
        
        # Kanserli görüntülerin dosya yollarını ve etiketlerini (1) ekle.
        for img_name in os.listdir(cancer_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(cancer_dir, img_name))
                self.labels.append(1)  # 1 -> Kanser
        
        # Normal (sağlıklı) görüntülerin dosya yollarını ve etiketlerini (0) ekle.
        for img_name in os.listdir(normal_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(normal_dir, img_name))
                self.labels.append(0)  # 0 -> Normal (Sağlıklı)
    
    def __len__(self):
        # Veri setindeki toplam görüntü sayısını döndür.
        return len(self.images)
    
    def __getitem__(self, idx):
        # Belirtilen index için görüntünün dosya yolunu al.
        img_path = self.images[idx]
        
        # Görüntüyü aç ve RGB formatına dönüştür (bazı resimler farklı modda olabilir).
        image = Image.open(img_path).convert('RGB')
        
        # İlgili görüntünün etiketini al (0 veya 1).
        label = self.labels[idx]
        
        # Eğer bir dönüşüm tanımlanmışsa, görüntüye uygula.
        if self.transform:
            image = self.transform(image)
            
        # Dönüştürülmüş görüntü ve etiket döndür.
        return image, label

def get_data_loaders(data_dir, batch_size=32):
    """
    Eğitim ve doğrulama (validation) veri yükleyicilerini oluşturan yardımcı fonksiyon.
    data_dir: Tüm görüntülerin bulunduğu üst dizin.
    batch_size: Kaç adet görüntünün bir seferde işleneceğini belirleyen parametre.
    """
    
    # Önce veri setini MedicalImageDataset sınıfımız ile oluşturuyoruz.
    dataset = MedicalImageDataset(data_dir)
    
    # Veri setinin %80'i eğitim, %20'si doğrulama için ayrılıyor.
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )
    
    # Eğitim veri yükleyicisi: Verileri karıştırarak (shuffle=True) okuyoruz.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Doğrulama veri yükleyicisi: Karıştırma kapalı (shuffle=False) varsayılan.
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Fonksiyon iki yükleyiciyi geri döndürüyor.
    return train_loader, val_loader
