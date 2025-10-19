import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# --- CNN architecture (same as training) ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# --- Load model and setup device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load("mnist_cnn_best.pth", map_location=device))
model.eval()
print(f"✅ Model loaded on {device}")

# --- MNIST test evaluation ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST('../data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)
print(f"\n📊 Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n")


# --- Upload & predict your own image ---
print("📂 Select an image to test (JPG/PNG)...")

# Open file dialog
Tk().withdraw()  # Hide the main Tkinter window
file_path = filedialog.askopenfilename(
    title="Select digit image",
    filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
)

if not file_path:
    print("⚠️ No file selected. Exiting.")
else:
    print(f"🖼️ Selected file: {file_path}")
    img = Image.open(file_path).convert("L")  # grayscale
    img = img.resize((28, 28))  # resize to MNIST format

    plt.imshow(img, cmap="gray")
    plt.title("Your Input Image")
    plt.axis("off")
    plt.show()

    # Convert image to tensor
    img_tensor = transform(img).unsqueeze(0).to(device)
    output = model(img_tensor)
    pred = output.argmax(dim=1, keepdim=True).item()
    print(f"🧠 Predicted digit: {pred}")
