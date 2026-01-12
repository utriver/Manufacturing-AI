import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import csv

# 1. Dataset 정의
class CatheterSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*')))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*')))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_paths[idx]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L')).astype(np.int64)
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].long()
        return img, mask

# 2. Transform
# albumentations에는 GaussianNoise가 없는 경우가 있습니다. 대신 Blur, RandomBrightnessContrast 등으로 대체합니다.
train_transform = A.Compose([
    A.Resize(512, 512),
    A.RandomRotate90(p=0.5),
    A.Blur(blur_limit=3, p=0.3),  # Blur로 대체
    A.RandomBrightnessContrast(p=0.3),  # 밝기/대비 변화 추가
    A.Normalize(),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2()
])

# 3. DataLoader
train_dataset = CatheterSegDataset(
    img_dir='./catheter_extrusion/train_data/image',
    mask_dir='./catheter_extrusion/train_data/mask',
    transform=train_transform
)
val_dataset = CatheterSegDataset(
    img_dir='./catheter_extrusion/valid_data/image',
    mask_dir='./catheter_extrusion/valid_data/mask',
    transform=val_transform
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 4. Model (smp.Unet, EfficientNet-b3 백본 사용 예시, 필요시 ResNet 등 교체 가능)
def get_model(n_classes=4):
    model = smp.Unet(
        encoder_name="efficientnet-b3",       # efficientnet-b3, resnet34, resnet50 등 선택 가능
        encoder_weights="imagenet",           # 또는 None
        in_channels=3,
        classes=n_classes,
        activation=None                       # logits (softmax 미포함)
    )
    return model.cuda()

model = get_model(n_classes=4)

# 5. Optimizer, Loss, Metric
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 200

def compute_iou(pred, target, n_classes=4):
    ious = []
    pred = torch.argmax(pred, dim=1)
    for cls in range(n_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

# 6. Training Loop (Best Model Save)
best_miou = 0
top5_mious = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs = imgs.cuda()
        masks = masks.cuda()
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = F.cross_entropy(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"[{epoch+1}] Train Loss: {train_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    val_loss = 0
    ious = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.cuda()
            masks = masks.cuda()
            outputs = model(imgs)
            loss = F.cross_entropy(outputs, masks)
            val_loss += loss.item()
            iou = compute_iou(outputs.cpu(), masks.cpu())
            ious.append(iou)
    val_miou = np.nanmean(ious)
    print(f"[{epoch+1}] Val Loss: {val_loss / len(val_loader):.4f}, mIoU: {val_miou:.4f}")

    # mIoU 상위 5개 관리
    top5_mious.append((epoch+1, val_miou))
    top5_mious = sorted(top5_mious, key=lambda x: x[1], reverse=True)[:5]

    if val_miou > best_miou:
        best_miou = val_miou
        torch.save(model.state_dict(), 'model_best_unet_200.pth')
        print(f"==> Best model saved! mIoU: {best_miou:.4f}")

# mIoU 상위 5개를 CSV로 저장
csv_path = 'top5_miou_unet_200.csv'
with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'miou'])
    for epoch_num, miou in top5_mious:
        writer.writerow([epoch_num, miou])
print(f"상위 5개 mIoU를 {csv_path}로 저장 완료")

torch.save(model.state_dict(), 'model_final_unet_200.pth')
print('마지막 에폭 모델 가중치(model_final_unet_200.pth) 저장 완료')
