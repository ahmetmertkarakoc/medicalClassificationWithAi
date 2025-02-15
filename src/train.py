import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from model import CancerDetectionModel
from data_loader import get_data_loaders
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

def train_model(args):
    # Cihazı (GPU/CPU) belirle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model örneğini oluştur ve cihaza (device) gönder
    model = CancerDetectionModel().to(device)
    
    # Binary Cross Entropy Loss (BCELoss) kullanıyoruz (çıkış 0 veya 1)
    criterion = nn.BCELoss()
    # Optimizasyon için Adam
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Eğitim ve doğrulama veri yükleyicilerini al
    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size)
    
    # Eğitim ve doğrulama kayıplarını (loss) tutmak için listeler
    train_losses = []
    val_losses = []
    
    # Belirtilen epoch sayısı kadar döngü
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()  # Modeli eğitim moduna al
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # tqdm ile eğitim ilerlemesini göster
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for images, labels in progress_bar:
            # Resimleri ve etiketleri cihaza (GPU/CPU) taşı
            images, labels = images.to(device), labels.float().to(device)
            
            # Geri yayılım (backprop) için sıfırlama
            optimizer.zero_grad()
            
            # Modelden çıkış al
            outputs = model(images).squeeze()
            
            # Kayıp (loss) hesapla
            loss = criterion(outputs, labels)
            
            # Geri yayılım ile ağırlıkları güncelle
            loss.backward()
            optimizer.step()
            
            # Toplam eğitim kaybını biriktir
            train_loss += loss.item()
            
            # 0.5 üstü değerleri kanser (1) olarak kabul et
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Her batch sonrası tqdm üzerinde anlık kaybı göster
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Validasyon aşaması (eval modunda)
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        # Validasyon sırasında gradyanlar hesaplanmıyor
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device)
                outputs = model(images).squeeze()
                
                # Validasyon kaybını ekle
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Tahmin doğruluğu hesapla
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Ortalama eğitim ve validasyon kayıpları
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        # Yüzde cinsinden doğruluk
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Epoch sonu istatistikleri
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Eğitim ve doğrulama kayıplarının grafiğini çizip kaydet
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_loss.png')
        
        # Epoch sonunda metrikleri hesapla
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)

        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
    
    # Eğitim ve doğrulama kayıplarının grafiğini çizip kaydet
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')

if __name__ == '__main__':
    # Komut satırından parametre alımı için ayarlar
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Eğitim verilerinin bulunduğu ana klasör')
    parser.add_argument('--model_path', type=str, default='models/cancer_detection_model.pth',
                        help='Kaydedilecek model dosya yolu')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Eğitim için kaç epoch kullanılacağı')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Kaç örneğin bir kerede işleneceği')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Öğrenme oranı')
    
    args = parser.parse_args()
    # Model eğitimini başlat
    train_model(args)
