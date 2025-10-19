import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary


# ---------------------------
# Model Definition
# ---------------------------
class Net(nn.Module):
    """Simple CNN for MNIST classification."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ---------------------------
# Training Function
# ---------------------------
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    
    for data, target in progress:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress.set_postfix({"loss": f"{running_loss / len(train_loader):.4f}"})
    
    avg_loss = running_loss / len(train_loader)
    print(f"🧠 Train Epoch {epoch}: Average Loss = {avg_loss:.4f}")
    return avg_loss


# ---------------------------
# Test Function
# ---------------------------
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"✅ Test Set: Avg Loss = {test_loss:.4f}, Accuracy = {accuracy:.2f}%")
    return test_loss, accuracy


# ---------------------------
# Main Execution
# ---------------------------
def main():
    # Hyperparameters
    batch_size = 64
    test_batch_size = 1000
    epochs = 25
    lr = 0.001
    seed = 42

    torch.manual_seed(seed)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"🔋 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # Model setup
    model = Net().to(device)
    summary(model, (1, 28, 28))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

    # Training loop
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        _, acc = test(model, device, test_loader)
        scheduler.step()

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "mnist_cnn_best.pth")
            print(f"💾 Model improved! Saved with accuracy {acc:.2f}%\n")

    print(f"🏁 Training complete. Best Accuracy: {best_acc:.2f}%")
    print("Model saved as mnist_cnn_best.pth")


if __name__ == "__main__":
    main()
