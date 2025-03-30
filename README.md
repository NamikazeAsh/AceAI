# AceAI

AceAI is a deep learning-based image classification model for playing cards. It uses EfficientNet as the backbone model and is trained on a dataset of playing cards to classify different card types accurately.

## Model Architecture

The model uses EfficientNet-B0 as the feature extractor, followed by a custom classification head:

```python
import timm
import torch.nn as nn

class AceAI(nn.Module):
    def __init__(self, num_classes=53):
        super(AceAI, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

## Training

The training loop uses cross-entropy loss and the Adam optimizer.

```python
import torch.optim as optim

num_epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AceAI(num_classes=53).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader.dataset)}")
```

## Prediction

To predict a playing card from an image:

```python
from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()
```

## Results Visualization

![image](https://github.com/user-attachments/assets/47c6d258-d53f-457b-83be-aa6a10aaabda)![image](https://github.com/user-attachments/assets/1987fc1d-bccd-4aba-8fe3-ef1fd85ac715)![image](https://github.com/user-attachments/assets/78c48f25-d79f-4ffb-8bcd-851115462360)![image](https://github.com/user-attachments/assets/42a8eb22-3421-4a98-951d-a395f9839edc)
