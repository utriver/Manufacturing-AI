## 🏥 AI 기반 의료기기(카테터 튜브) 제조 정밀도 향상 및 토출 각도 추정 연구
'''

의료기기 제조 공정의 수작업 측정 한계를 극복하고 생산성을 높이기 위해, 딥러닝 분할(Segmentation) 모델과 고도화된 각도 추정 알고리즘을 결합한 지능화 솔루션입니다.
(제조 인공지능 응용 연구 텀 프로젝트)

'''

##📝 프로젝트 개요 (Project Overview)

카테터 튜브 제조 시 수지 토출 각도의 정확한 추정은 제품의 품질과 안정성에 직결됩니다. 본 프로젝트는 영상 처리를 통해 토출 영역을 정밀하게 분할하고, 기존 PCA(주성분 분석) 방식의 오작동 문제를 해결하는 새로운 중심점 탐색 및 방향 보정 알고리즘을 제안하여 제조 정밀도를 혁신적으로 개선했습니다.

🚀 주요 특징 (Key Features)

1. 제조 공정 지능화 (AI for Manufacturing)

수동 측정 한계 극복: 인적 오류(Human Error)를 최소화하고 실시간 각도 추정 및 조정 가능 환경 구축.

품질 일관성 확보: 대량의 제조 데이터를 일관된 기준으로 처리하여 의료기기 안전성 제고.

2. 고도화된 각도 추정 알고리즘 (Advanced Angle Estimation)

기존 PCA 기반 추정 방식이 Segmentation 결과에 따라 중심점이 이탈하거나 방향이 반대로 측정되는 문제를 해결하기 위해 다음과 같은 로직을 구현했습니다.

중심점 탐색 알고리즘:

원본 이미지 내부에서 토출 구멍에 해당하는 특정 픽셀 범위를 지정.

해당 범위 내 클래스(2, 3) 픽셀 중 거리가 가장 짧은 구간의 중점을 토출 각도의 실제 중심점으로 재지정하여 정확도 향상.

PCA 방향성 보정:

벡터의 방향성이 일관되지 않는 문제를 해결하기 위해, "중심점으로부터 더 먼 지점이 벡터의 진행 방향"이라는 조건을 설정하여 방향성 오류를 해결함.

3. 분할 학습 모델 (Segmentation Model)

데이터 전처리: 제조 환경 이미지를 클래스별(Catheter, Hole 등)로 라벨링 및 마스킹 수행.

성능 지표: mIOU(mean Intersection over Union)를 활용하여 모델의 분할 정확도를 정량적으로 관리함.

반복 학습: Epoch 최적화를 통해 복잡한 토출 패턴에서도 높은 분할 성능을 확보함.

🛠 기술 스택 (Technical Stack)

분류

상세 내용

AI & Deep Learning

Semantic Segmentation (U-Net/DeepLabV3+), PyTorch/TensorFlow

Computer Vision

OpenCV, PCA (Principal Component Analysis), Pixel-level Analysis

Data Analysis

mIOU Metrics, Pixel Distance Calculation

Programming

Python, Numpy, Matplotlib

Tools

Git, Google Colab/Jupyter Notebook

📈 연구 성과 (Research Results)

알고리즘 정밀도 개선: 기존 PCA 방식 대비 중심점 오차 및 벡터 방향성 오류를 획기적으로 감소시킴.

제조 공정 적용 가능성 확인: 복잡한 수지 토출 형상에서도 안정적인 각도 데이터를 산출하여 자동화 공정 투입 기반을 마련함.

학기말 최종 구현 성공: 데이터 전처리부터 모델 학습, 최종 각도 추정 로직 구현까지 전 과정을 완수함.

📂 프로젝트 구조 (Structure)

├── data_preprocessing/     # 라벨링 데이터 및 마스크 이미지 생성 스크립트
├── models/                # Segmentation 학습 모델 아키텍처 및 가중치 파일
├── inference/             # 각도 추정 및 방향 보정 알고리즘 소스 코드
└── notebooks/             # 학습 과정 및 성능 지표(mIOU) 확인용 Jupyter Notebook


💡 실행 방법 (Usage)

사전 준비: requirements.txt에 명시된 라이브러리(OpenCV, PyTorch 등)를 설치합니다.

모델 로드: /models 디렉토리의 최적 가중치 파일을 불러옵니다.

각도 추정 실행:

python estimate_angle.py --input [image_path]


결과 확인: 보정된 중심점과 방향 벡터가 포함된 결과 이미지가 /results 폴더에 생성됩니다.
