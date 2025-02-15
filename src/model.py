from torch import nn
import torchvision.models as models

class CancerDetectionModel(nn.Module):
    def __init__(self, pretrained=True):
        super(CancerDetectionModel, self).__init__()
        # ResNet18'i önceden eğitilmiş ağırlıklarla yükle
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Son katmanı değiştir (binary classification için)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.resnet(x) 