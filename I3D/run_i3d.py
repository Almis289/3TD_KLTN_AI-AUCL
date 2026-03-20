import ssl
import certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import torch
import torchvision.transforms as transforms
import cv2

# Load model
model = torch.hub.load(
    'facebookresearch/pytorchvideo',
    'slow_r50',
    pretrained=True,
    skip_validation=True
)
model.eval()

# Transform chuẩn
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load video
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("❌ Không mở được video:", video_path)
    exit()

frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame)
    frames.append(frame)

cap.release()

print("Số frame đọc được:", len(frames))

if len(frames) == 0:
    raise ValueError("❌ Không có frame nào → kiểm tra lại video!")

# Convert tensor
video = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(video)

print("Prediction:", output.argmax().item())