import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
from torchvision import transforms
import timm
import torch.nn as nn

# Initialize Flask app
app = Flask(__name__)

# Define your model architecture
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
        output = self.classifier(x)
        return output

# Initialize the model
model = AceAI(num_classes=53)

# Load the state_dict (weights)
model.load_state_dict(torch.load("AceAI.pth", map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Match the size your model was trained with
    transforms.ToTensor(),
])

# The card dictionary mapping index to card name
card_dict = {
    0: 'A♣️', 1: 'A♦️', 2: 'A♥️', 3: 'A♠️',
    4: '8♣️', 5: '8♦️', 6: '8♥️', 7: '8♠️',
    8: '5♣️', 9: '5♦️', 10: '5♥️', 11: '5♠️',
    12: '4♣️', 13: '4♦️', 14: '4♥️', 15: '4♠️',
    16: 'J♣️', 17: 'J♦️', 18: 'J♥️', 19: 'J♠️',
    20: '🃏',
    21: 'K♣️', 22: 'K♦️', 23: 'K♥️', 24: 'K♠️',
    25: '9♣️', 26: '9♦️', 27: '9♥️', 28: '9♠️',
    29: 'Q♣️', 30: 'Q♦️', 31: 'Q♥️', 32: 'Q♠️',
    33: '7♣️', 34: '7♦️', 35: '7♥️', 36: '7♠️',
    37: '6♣️', 38: '6♦️', 39: '6♥️', 40: '6♠️',
    41: '10♣️', 42: '10♦️', 43: '10♥️', 44: '10♠️',
    45: '3♣️', 46: '3♦️', 47: '3♥️', 48: '3♠️',
    49: '2♣️', 50: '2♦️', 51: '2♥️', 52: '2♠️'
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    image_file = request.files["image"]
    image = Image.open(io.BytesIO(image_file.read()))

    # Ensure the image is in RGB mode (remove alpha channel if it exists)
    image = image.convert("RGB")

    image = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    # Map the predicted class to the card name
    predicted_card = card_dict.get(predicted.item(), "Unknown Card")

    return jsonify({"prediction": predicted_card})

if __name__ == "__main__":
    app.run(debug=True)
