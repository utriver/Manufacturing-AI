# 🏥 AI 기반 의료기기(카테터 튜브) 제조 정밀도 향상 및 토출 각도 추정 연구

> **Manufacturing AI Application Research Term Project**
> 딥러닝 Segmentation과 고도화된 각도 추정 알고리즘을 결합한 지능화 솔루션

---

## 📝 프로젝트 개요 (Project Overview)
<img width="512" height="517" alt="image" src="https://github.com/user-attachments/assets/d1e3f3a5-795e-4369-b14a-ccc102970197" />

카테터 튜브 제조 시 수지 토출 각도의 정확한 추정은 제품의 품질과 안정성에 직결됩니다. 본 프로젝트는 영상 처리를 통해 토출 영역을 정밀하게 분할하고, 기존 **PCA(주성분 분석) 방식의 오작동 문제를 해결**하는 새로운 중심점 탐색 및 방향 보정 알고리즘을 제안하여 제조 정밀도를 혁신적으로 개선했습니다.

---

## 🚀 주요 특징 (Key Features)

### 1. 제조 공정 지능화 (AI for Manufacturing)
* **수동 측정 한계 극복:** 인적 오류(Human Error)를 최소화하고 실시간 각도 추정 및 조정 가능 환경 구축
* **품질 일관성 확보:** 대량의 제조 데이터를 일관된 기준으로 처리하여 의료기기 안전성 제고

### 2. 고도화된 각도 추정 알고리즘 (Advanced Angle Estimation)
기존 PCA 기반 추정 방식의 중심점 이탈 및 방향 반전 문제를 해결하기 위해 다음 로직을 구현했습니다.

* **중심점 탐색 알고리즘:**
  - 원본 이미지 내부에서 토출 구멍에 해당하는 특정 픽셀 범위를 지정
  - 해당 범위 내 클래스(2, 3) 픽셀 중 거리가 가장 짧은 구간의 중점을 실제 중심점으로 재지정하여 정확도 향상
* **PCA 방향성 보정:**
  - 벡터의 방향성이 일관되지 않는 문제를 해결하기 위해, **"중심점으로부터 더 먼 지점이 벡터의 진행 방향"**이라는 조건을 설정하여 방향성 오류 해결

### 3. 분할 학습 모델 (Segmentation Model)
* **성능 지표:** `mIOU(mean Intersection over Union)`를 활용하여 모델의 분할 정확도를 정량적으로 관리
* **반복 학습:** `Epoch` 최적화를 통해 복잡한 토출 패턴에서도 높은 분할 성능 확보

---

## 🛠 기술 스택 (Technical Stack)

| 구분 | 기술 스택 |
|:---:|:---|
| **AI & DL** | `Semantic Segmentation (U-Net/DeepLabV3+)`, `PyTorch`, `TensorFlow` |
| **CV** | `OpenCV`, `PCA (Principal Component Analysis)`, `Pixel-level Analysis` |
| **Analysis** | `mIOU Metrics`, `Pixel Distance Calculation` |
| **Language** | `Python (Numpy, Matplotlib)` |

---

## 📈 연구 성과 (Research Results)
* **알고리즘 정밀도 개선:** 기존 PCA 방식 대비 중심점 오차 및 벡터 방향성 오류를 획기적으로 감소
* **제조 공정 적용 가능성 확인:** 복잡한 수지 토출 형상에서도 오차율 1%이내의 안정적인 각도 데이터 산출, 자동화 공정 투입 기반 마련
* **전 과정 구현 성공:** 데이터 전처리부터 모델 학습, 최종 각도 추정 로직 구현까지 프로세스 완수

---

## 📂 프로젝트 구조 (Structure)
```text
├── model_best_unet_200.pth: 가중치 파일
├── predictangle_Unet.py: mask추정 및 각도 추정
├── train_3_unet.py: 분할 학습 모델
```
## 💡 실행 방법 (Usage)
* **설치 라이브러리:** 
```
pip install numpy pandas pillow matplotlib tqdm albumentations segmentation-models-pytorch torch torchvision scikit-learn
```

* **프로그램 실행**
1. 학습 프로그램인  train_3_unet.py를 실행시키면 가중치 파일인 model_best_unet_200.pth이 생성
2. predictangle_Unet.py를 실행시키면 예측각도에 대한 csv파일 생성 및 각 이미지 생성이 됨
