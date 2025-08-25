import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Avoid crash on broken images

# Custom transform to ensure all images are RGB
class ConvertToRGB:
    def __call__(self, img):
        return img.convert("RGB")

def load_dataset(data_path, batch_size=32):
    # ✅ Print diagnostic info
    print(f"📁 Scanning dataset path: {data_path}\n")
    for root, dirs, files in os.walk(data_path):
        print(f"🔍 {root}: {len(files)} files")

    # ✅ Check if folders contain at least one supported image
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    empty_classes = []
    for class_dir in os.listdir(data_path):
        full_class_path = os.path.join(data_path, class_dir)
        if os.path.isdir(full_class_path):
            valid_files = [f for f in os.listdir(full_class_path) if f.lower().endswith(valid_extensions)]
            if not valid_files:
                empty_classes.append(class_dir)

    if empty_classes:
        print("⚠️ Warning: The following class folders contain no valid image files:")
        for cls in empty_classes:
            print(f" - {cls}")
        raise FileNotFoundError("❌ No valid images found in some class folders. Please check your dataset.")

    # ✅ Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        ConvertToRGB(),              # Ensures all images are RGB
        transforms.ToTensor(),
    ])

    # ✅ Load dataset
    dataset = datasets.ImageFolder(data_path, transform=transform)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # ✅ Use num_workers=0 for Windows
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    print(f"✅ Dataset loaded: {len(dataset)} total images across {len(dataset.classes)} classes")
    return train_loader, val_loader, len(dataset.classes)
