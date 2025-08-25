import os
import time
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
from dataset_loader import load_dataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import tkinter as tk
from PIL import Image, ImageTk

# Device
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    DEVICE = torch.device("cuda")
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ CUDA not available, using CPU instead")

# Output paths
output_dir = r"C:\Users\priya\OneDrive\Documents\Federated learning output"
os.makedirs(output_dir, exist_ok=True)

plot_path = os.path.join(output_dir, "1. Plot Training Accuracy_Loss per Round.png")
csv_path = os.path.join(output_dir, "4. Training_Metrics.csv")
log_path = os.path.join(output_dir, "3. Evaluate on Validation Set and Print Accuracy.txt")
model_path = os.path.join(output_dir, "2. Save the Final Trained Model.pth")

# Load data
print("📦 Loading dataset...")
train_loader, val_loader, num_classes = load_dataset(
    r"D:\My files\My Projects\Solar panel research\Dataset\sorted_images\train",
    batch_size=32
)
print("✅ Dataset loaded.")

# Load pretrained model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Metrics
loss_list = []
acc_list = []

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.train()
        print("🚀 Training started...")
        total_loss = 0.0
        sample_count = 0
        max_samples = 1000

        for i, (images, labels) in enumerate(train_loader):
            if sample_count >= max_samples:
                break
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            sample_count += len(images)

        avg_loss = total_loss / (sample_count / train_loader.batch_size)
        loss_list.append(avg_loss)
        print(f"✅ Training complete | Avg Loss: {avg_loss:.4f}")
        return self.get_parameters({}), sample_count, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        model.eval()
        print("🔍 Evaluating...")
        correct = total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        acc_list.append(accuracy)
        print(f"📈 Eval Accuracy: {accuracy:.2%}")
        return float(1.0 - accuracy), total, {"accuracy": accuracy}

# ✅ Plot display (with persistent reference)
def show_plot():
    if not os.path.exists(plot_path):
        print("⚠️ No plot to display.")
        return

    root = tk.Tk()
    root.title("Training Accuracy & Loss")

    img = Image.open(plot_path)
    tk_img = ImageTk.PhotoImage(img)

    label = tk.Label(root, image=tk_img)
    label.image = tk_img
    label.pack()

    root.mainloop()

# ✅ Save all outputs
def save_results():
    if not acc_list:
        print("⚠️ No accuracy data to save.")
        return

    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to: {model_path}")

    with open(log_path, "w") as f:
        for i, acc in enumerate(acc_list, 1):
            f.write(f"Round {i}: Accuracy = {acc:.4f}\n")
    print(f"📝 Accuracy log saved: {log_path}")

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Accuracy", "Loss"])
        for i in range(max(len(acc_list), len(loss_list))):
            acc = acc_list[i] if i < len(acc_list) else ""
            loss = loss_list[i] if i < len(loss_list) else ""
            writer.writerow([i + 1, acc, loss])
    print(f"📊 CSV saved: {csv_path}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc_list, marker='o')
    plt.title("Validation Accuracy per Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(loss_list, marker='x', color='red')
    plt.title("Training Loss per Round")
    plt.xlabel("Round")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"📈 Plot saved: {plot_path}")

# ✅ Main
def main():
    client = FlowerClient()
    print("🌐 Connecting to server...")
    start = time.time()

    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client()
    )

    print(f"⏱️ Total time: {time.time() - start:.2f}s")
    save_results()
    show_plot()

if __name__ == "__main__":
    main()
