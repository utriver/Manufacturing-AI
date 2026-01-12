import os
import glob
import numpy as np
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def get_midpoint_between_closest_outer_points_in_box(mask, box_left_top, box_right_bottom):
    ys_r, xs_r = np.where(mask == 2)
    ys_b, xs_b = np.where(mask == 3)
    red_coords = np.stack([xs_r, ys_r], axis=1)
    blue_coords = np.stack([xs_b, ys_b], axis=1)
    def in_box(pt):
        x, y = pt
        return (box_left_top[0] <= x <= box_right_bottom[0]) and (box_left_top[1] <= y <= box_right_bottom[1])
    red_coords_in = np.array([pt for pt in red_coords if in_box(pt)])
    blue_coords_in = np.array([pt for pt in blue_coords if in_box(pt)])
    if len(red_coords_in) == 0 or len(blue_coords_in) == 0:
        return None, None, None, red_coords_in, blue_coords_in
    dists = cdist(red_coords_in, blue_coords_in)
    idx = np.unravel_index(np.argmin(dists), dists.shape)
    closest_red = red_coords_in[idx[0]]
    closest_blue = blue_coords_in[idx[1]]
    center = (closest_red + closest_blue) / 2
    return center, closest_red, closest_blue, red_coords_in, blue_coords_in

def get_resin_centroid(mask):
    ys, xs = np.where(mask == 1)
    if len(xs) < 2:
        return None
    centroid = np.array([xs.mean(), ys.mean()])
    return centroid

def pca_vector_direction(points, center=None):
    if len(points) < 2:
        return None
    pts = points - points.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_vec = eigvecs[:, np.argmax(eigvals)]
    # 중심점 기준 방향 보정
    if center is not None:
        dists = np.linalg.norm(points - center, axis=1)
        farthest_idx = np.argmax(dists)
        farthest_vec = points[farthest_idx] - center
        # 방향 보정 (내적 < 0이면 뒤집음)
        if np.dot(main_vec, farthest_vec) < 0:
            main_vec = -main_vec
    return main_vec

def get_model(n_classes=4):
    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights=None,
        in_channels=3,
        classes=n_classes,
        activation=None
    )
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(n_classes=4).to(device)
model.load_state_dict(torch.load('model_best_unet_200.pth', map_location=device))
model.eval()

test_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2()
])

all_results = []

for case_idx in range(9):
    case_name = f'case{case_idx}'
    test_image_dir = f'./catheter_extrusion/test_data/{case_name}/IMAGE'
    save_dir = f'./angle_visualize_entrybox_unet_200/{case_name}'
    os.makedirs(save_dir, exist_ok=True)
    img_paths = sorted(glob.glob(os.path.join(test_image_dir, '*')))

    for img_path in img_paths:
        img = np.array(Image.open(img_path).convert('RGB'))
        orig_size = (img.shape[1], img.shape[0])
        transformed = test_transform(image=img)
        input_tensor = transformed['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred_mask_resized = Image.fromarray(pred_mask).resize(orig_size, Image.NEAREST)
        pred_mask_resized = np.array(pred_mask_resized)

        box_left_top = (350, 350)
        box_right_bottom = (510, 510)
        mid_entry, closest_red, closest_blue, red_coords_in, blue_coords_in = get_midpoint_between_closest_outer_points_in_box(
            pred_mask_resized, box_left_top, box_right_bottom
        )

        # PCA 방향(중심점 기반 방향 보정)
        red_vec = pca_vector_direction(red_coords_in, center=mid_entry)
        blue_vec = pca_vector_direction(blue_coords_in, center=mid_entry)

        # 평균 방향 벡터 → 각도(도 단위, 0~360)
        if red_vec is not None and blue_vec is not None:
            mean_vec = (red_vec + blue_vec) / 2
            mean_vec = mean_vec / np.linalg.norm(mean_vec)
            angle_deg = (np.degrees(np.arctan2(mean_vec[1], -mean_vec[0]))+180) % 360
        elif red_vec is not None:
            red_vec = red_vec / np.linalg.norm(red_vec)
            angle_deg = (np.degrees(np.arctan2(red_vec[1], -red_vec[0]))+180) % 360
        elif blue_vec is not None:
            blue_vec = blue_vec / np.linalg.norm(blue_vec)
            angle_deg = (np.degrees(np.arctan2(blue_vec[1], -blue_vec[0]))+180) % 360
        else:
            angle_deg = None


        base_name = os.path.splitext(os.path.basename(img_path))[0]
        all_results.append([case_name, base_name, angle_deg])

        # 시각화
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        if len(red_coords_in) > 0:
            plt.scatter(red_coords_in[:, 0], red_coords_in[:, 1], s=8, c='red', marker='o', label='red outline')
        if len(blue_coords_in) > 0:
            plt.scatter(blue_coords_in[:, 0], blue_coords_in[:, 1], s=8, c='blue', marker='o', label='blue outline')
        if mid_entry is not None and closest_red is not None and closest_blue is not None:
            plt.plot([closest_red[0], closest_blue[0]], [closest_red[1], closest_blue[1]],
                     color='yellow', linewidth=2, linestyle='--', label='min dist')
            plt.scatter([mid_entry[0]], [mid_entry[1]], s=80, color='green', label='midpoint')
        # PCA 평균방향 화살표
        if ((red_vec is not None) or (blue_vec is not None)) and (mid_entry is not None):
            center = mid_entry
            if red_vec is not None and blue_vec is not None:
                vec = mean_vec
            elif red_vec is not None:
                vec = red_vec
            else:
                vec = blue_vec
            plt.arrow(center[0], center[1], vec[0]*90, vec[1]*90, color='green', width=5, head_width=20, label='mean angle')
            plt.title(f'{case_name}_{base_name}: PCA angle={angle_deg:.2f}°')
        else:
            plt.title(f'{case_name}_{base_name}: No valid outline')
        plt.gca().add_patch(
            plt.Rectangle(box_left_top,
                          box_right_bottom[0] - box_left_top[0],
                          box_right_bottom[1] - box_left_top[1],
                          linewidth=2, edgecolor='lime', facecolor='none', label='box')
        )
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{base_name}_entrybox_pca_angle.png'))
        plt.close()

# 전체를 한번에 저장
df = pd.DataFrame(all_results, columns=['case', 'filename', 'angle_deg'])
df.to_csv('./angle_visualize_entrybox_unet_200/all_angle_results_pca.csv', index=False)
print(f"모든 case test 이미지 결과를 all_angle_results_pca.csv로 한 번에 저장 완료!")
