import torch
from torchvision import transforms
from PIL import Image
from model import CancerDetectionModel
import argparse

def test_single_image(image_path, model_path):
    # PyTorch cihaz (device) seçimi: GPU varsa 'cuda', yoksa 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Modeli oluştur ve kaydedilen parametreleri yükle
    model = CancerDetectionModel()
    model.load_state_dict(torch.load(model_path))  # Eğitilen ağırlıkları model'e yükle
    model.to(device)                               # Modeli kullanılacak cihaza (CPU veya GPU) gönder
    model.eval()                                   # Modeli değerlendirme moduna geçir (dropout, batchnorm vb. devre dışı)
    
    # Görüntü dönüşümleri (boyutlandırma, tensöre dönüştürme, normalizasyon)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Resmi aç, RGB'ye dönüştür, dönüşümleri uygula ve batch boyutunu 1 yap
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Model ile tahmin yap
    with torch.no_grad():           # Geri yayılım (backprop) kapalı
        output = model(image_tensor)
        probability = output.item() * 100  # Çıkış 0-1 arası, %'ye dönüştür
    
    # Sonuç analizini genişlet
    confidence_level = "Yüksek" if abs(probability - 50) > 25 else "Orta" if abs(probability - 50) > 10 else "Düşük"
    
    print(f"\nDetaylı Sonuç Analizi:")
    print(f"Görüntü: {image_path}")
    print(f"Kanser Olma Olasılığı: {probability:.2f}%")
    print(f"Tahmin: {'Kanser' if probability > 50 else 'Normal'}")
    print(f"Tahmin Güven Seviyesi: {confidence_level}")
    print(f"Not: Bu sonuçlar sadece referans amaçlıdır ve klinik teşhis için kullanılmamalıdır.")

if __name__ == '__main__':
    # Komut satırından parametre alma ayarları
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True,
                        help='Test edilecek görüntünün yolu')
    parser.add_argument('--model_path', type=str, default='models/cancer_model.pth',
                        help='Eğitilmiş model dosyasının yolu')
    
    args = parser.parse_args()
    
    # Seçilen resim ve model ile test fonksiyonunu çağır
    test_single_image(args.image_path, args.model_path)
