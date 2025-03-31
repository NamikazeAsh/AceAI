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
    0: 'ace of clubs', 1: 'ace of diamonds', 2: 'ace of hearts', 3: 'ace of spades', 
    4: 'eight of clubs', 5: 'eight of diamonds', 6: 'eight of hearts', 7: 'eight of spades', 
    8: 'five of clubs', 9: 'five of diamonds', 10: 'five of hearts', 11: 'five of spades', 
    12: 'four of clubs', 13: 'four of diamonds', 14: 'four of hearts', 15: 'four of spades', 
    16: 'jack of clubs', 17: 'jack of diamonds', 18: 'jack of hearts', 19: 'jack of spades', 
    20: 'joker', 21: 'king of clubs', 22: 'king of diamonds', 23: 'king of hearts', 
    24: 'king of spades', 25: 'nine of clubs', 26: 'nine of diamonds', 27: 'nine of hearts', 
    28: 'nine of spades', 29: 'queen of clubs', 30: 'queen of diamonds', 31: 'queen of hearts', 
    32: 'queen of spades', 33: 'seven of clubs', 34: 'seven of diamonds', 35: 'seven of hearts', 
    36: 'seven of spades', 37: 'six of clubs', 38: 'six of diamonds', 39: 'six of hearts', 
    40: 'six of spades', 41: 'ten of clubs', 42: 'ten of diamonds', 43: 'ten of hearts', 
    44: 'ten of spades', 45: 'three of clubs', 46: 'three of diamonds', 47: 'three of hearts', 
    48: 'three of spades', 49: 'two of clubs', 50: 'two of diamonds', 51: 'two of hearts', 
    52: 'two of spades'
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
