# 얼굴·지문 특징 추출기

사전학습 가중치를 고정한 추론 모듈입니다. 두 모델 모두 이미지 한 장을
`float32`, `(512,)`, L2 norm 약 1인 임베딩으로 변환합니다.
학습, Thresholding Model, 임베딩 이진화, BCH/FE 연결은 포함하지 않습니다.

| 폴더 | 모델 | 실행 방식 | 출력 |
|---|---|---|---|
| `arcface_iresnet50/` | InsightFace ArcFace IResNet50, `w600k_r50` | 공식 ONNX 가중치 + ONNX Runtime | 512 |
| `deepprint_flx/` | flx `DeepPrint_TexMinu_512` | PyTorch, `eval()`, 모든 파라미터 고정 | texture 256 + minutiae 256 |

각 폴더의 `vendor/`에는 실행에 필요한 출처 소스와 라이선스가 있습니다.
사전학습 가중치는 `weights/`에 별도로 준비하며 공개 저장소에는 포함하지 않습니다.
`SOURCES.json`에 버전, 다운로드 주소, SHA-256 및 로컬 수정 내용을 기록했습니다.
`smoke_test.py`가 전체 데이터에서 동작을 검사하고 `results/`에 파생 결과를 저장합니다.

## 2026-10-29 확인 결과

RTX 3060에서 고정된 사전학습 가중치로 전체 원본을 추론했습니다.

| 데이터 | 원본 수 | 성공 | 실패 | 저장 임베딩 형태 |
|---|---:|---:|---:|---|
| MUCT, 276명 | 3,755 | 3,755 | 0 | `(3755, 512)` |
| U.are.U, 7명·65손가락 | 520 | 520 | 0 | `(520, 512)` |

NaN/Inf 없음, L2 norm 약 1, 서로 다른 입력에서 비상수 출력,
원본 파일 해시 보존을 확인했습니다. 반복 입력의 최대 절대 차이는 얼굴 0,
지문 약 `2.24e-8`이었습니다. 얼굴 전체에서 자동 검출이 성공했고 다중 얼굴 검출은 없었습니다.
전체 실행 기록은 원실험 환경의 `results/full_dataset_check/report.json`,
데이터별 임베딩과 파일별 처리 기록은 같은 폴더에 보관했습니다. 이 결과 폴더는 공개 저장소에서 제외합니다.

시스템의 `nvidia-smi`는 NVML 드라이버/라이브러리 버전 불일치를 보고했습니다.
실제 PyTorch CUDA 연산과 ONNX CUDA 추론은 위 전체 검사에서 성공했습니다.
GPU 드라이버나 시스템 라이브러리 변경은 수행하지 않았습니다.

## 실행 환경

현재 프로젝트의 `.venv`는 `/home/uxm2/miniconda/envs/deepfake`의 Python 3.10 및
기존 PyTorch 2.5.1+cu121을 읽기 전용으로 재사용하는 `--system-site-packages` 환경입니다.
추가 ONNX/영상 처리 패키지는 이 프로젝트의 `.venv`에만 설치했습니다.
기존 Conda 환경이나 GPU 드라이버는 변경하지 않았습니다.

`run.sh`는 프로세스에만 CUDA 라이브러리 경로를 설정합니다. GPU 실행이 필요하면
GPU 장치에 접근할 수 있는 일반 터미널에서 실행합니다. `--device cuda`가 지정되면
GPU를 사용할 수 없을 때 CPU로 조용히 전환하지 않고 오류를 보고합니다.

```bash
cd /home/uxm2/hsh/2026_2/multimodal_biometric

# 두 모달리티 각각 처음 8장으로 빠르게 확인
bash feature_extractors/run.sh -m feature_extractors.smoke_test --device cuda --limit 8

# 원본 전체: MUCT 3,755장 + U.are.U 520장
bash feature_extractors/run.sh -m feature_extractors.smoke_test --device cuda

# CPU에서도 실행 가능
bash feature_extractors/run.sh -m feature_extractors.smoke_test --device cpu --limit 2
```

새 환경을 구성할 때는 Python 3.10 이상에서 PyTorch/CUDA 버전을 먼저 맞춥니다.
아래 명령은 참고용이며, 현재 환경을 재생성할 필요는 없습니다.

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r feature_extractors/requirements.txt
```

## 코드에서 사용

프로젝트 루트를 Python 모듈 경로에 둡니다. 스크립트를 실행할 때
`bash feature_extractors/run.sh your_script.py`를 사용하면 됩니다.

```python
from feature_extractors.arcface_iresnet50 import ArcFaceIResNet50
from feature_extractors.deepprint_flx import DeepPrintFlx

face = ArcFaceIResNet50(device="cuda")
fingerprint = DeepPrintFlx(device="cuda")

z_face = face.extract("muct-master/extracted/jpg/i000qa-fn.jpg")
z_finger = fingerprint.extract("UareU/012_1_1.tif")
# 각각 shape=(512,), dtype=float32. 두 예시는 실제 동일인을 뜻하지 않습니다.

batch = fingerprint.extract_batch(["UareU/012_1_1.tif", "UareU/012_1_2.tif"])
# shape=(2, 512)
```

가중치가 없거나 호환되지 않으면 실패합니다. 무작위 초기화 모델을 대신 사용하지 않습니다.
`extract()`는 임베딩만 반환하고 사용자 ID를 입력받지 않습니다.

## 얼굴 입력 처리

1. SCRFD `det_10g`로 얼굴과 5개 랜드마크를 자동 검출합니다. 검출 입력은 640×640,
   score threshold는 0.5입니다.
2. 공식 ArcFace 기준점에 similarity transform으로 정렬하여 112×112 RGB 입력을 만듭니다.
3. `(pixel - 127.5) / 127.5`로 정규화하고 `w600k_r50`을 실행합니다.
4. 512차원 원시 임베딩을 L2 정규화합니다.

MUCT의 수동 랜드마크 CSV는 사용하지 않았습니다. 자동 검출 실패는 실패로 기록하며
수동 정렬로 대체하지 않습니다. 얼굴이 여러 개 검출되면 가장 큰 bounding box를 선택하고
검출 개수와 선택 결과를 manifest에 기록합니다.

공개 코드에는 ONNX 추론에 필요한 vendor 소스를 선별했습니다. 실제 실행 가중치는
InsightFace가 공개한 `buffalo_l` 패키지의 `w600k_r50.onnx`입니다. 임의로 PyTorch 모델을
초기화하거나 별도 학습을 수행하지 않습니다.

## 지문 입력 처리

원본 TIFF는 520장 모두 팔레트 모드 `P`, 가로 326×세로 357, 단일 프레임입니다.
팔레트 인덱스를 밝기로 해석하지 않고 PIL `convert("L")`로 회색조를 얻습니다.

`flx`에는 U.are.U 전용 loader가 없으므로, 공개 코드의 광학 센서 경로에 맞춰 다음
고정 처리를 적용합니다.

1. 회색조 값을 `[0, 1]`로 변환합니다.
2. 흰색으로 좌우 15/16픽셀을 채워 357×357 정사각형으로 만듭니다. 원본 영역을 자르지 않습니다.
3. antialias를 사용해 299×299로 크기를 조절합니다.
4. upstream의 Gabor 기반 지문 영상 전처리기를 적용합니다. `ridge_width=3.8`은
   MCYT optical 기본값을 가져온 것으로, U.are.U에서 최적화하거나 성능 검증한 값이 아닙니다.
5. 고정된 DeepPrint에서 각각 L2 정규화된 texture/minutiae 256차원을 얻어 이 순서로
   결합하고 다시 L2 정규화합니다. 투영이나 차원 축소는 없습니다.

여기서 영상의 Gabor 이진화는 나중에 임베딩을 scalar tau로 비트열로 만드는 과정과 다릅니다.
DeepPrint의 출력은 계속 실수 벡터입니다. 저장된 training classifier는 로드 일치 검사에만
포함되고 추론에서는 사용하지 않습니다. 사용자·손가락 ID를 모델에 전달하지 않습니다.

Upstream texture branch는 이미 `(N,D)`인 Linear 출력에 `torch.squeeze`를 적용하여
N=1에서 실패했습니다. 이 불필요한 squeeze만 제거했습니다. N>1의 계산식과 가중치는
동일하며, 단일 이미지와 배치 입력의 결과 일관성을 검사합니다.

## 검사와 산출물

각 실행은 별도 출력 폴더를 생성합니다. 기존 결과 폴더를 덮어쓰지 않습니다.

- `report.json`: 입력 수, 출력 크기, 장치, 모델 해시, 수치 검사 및 실패 수.
- `face_manifest.csv`, `fingerprint_manifest.csv`: 모든 입력의 파일명, 데이터 관리용 ID,
  원본 모드/크기/SHA-256, 성공·실패, 임베딩 행 번호.
- `face_embeddings.npz`, `fingerprint_embeddings.npz`: 성공한 입력의 `embeddings`와
  `files`. `np.load(..., allow_pickle=False)`로 읽습니다.

입력 전후 파일 SHA-256 일치, 512차원, 유한값, L2 norm, 반복 추론, 서로 다른 입력의
출력 비상수성을 검사합니다. 지문은 단일/배치 추론 일관성과 모델 고정 상태도 확인합니다.

얼굴 사람 ID는 파일명에서 읽으며, 비연속 ID를 재해석하지 않습니다. suffix의 예외값을
그대로 보관하고 파일을 제외하지 않습니다. 반전본 `ir...`은 원본 집계에서 별도로 구분합니다.
지문 manifest는 사람 ID와 손가락 ID를 별도 열에 보관하여 7명/65손가락을 구분합니다.
촬영 번호를 날짜·세션으로 해석하지 않습니다.

이 검사는 추론 파이프라인의 정상 동작 확인입니다. 같은 사람/다른 사람 인증 정확도,
FAR/FRR, FE 키 복구율을 측정하지 않았고 학습·검증·시험 분할도 생성하지 않았습니다.
두 데이터셋 사이의 가상 사용자 대응도 만들지 않았습니다.

## 출처

- InsightFace: https://github.com/deepinsight/insightface/tree/3e6486942a1be2da0e5b475fac375ea73264bd21
  (코드 MIT; 공개 사전학습 가중치는 비상업 연구용, upstream 안내 참조).
- flx: https://github.com/tim-rohwedder/fixed-length-fingerprint-extractors/tree/7accfca1f33b9b42bfd220e43cd5bc13b4a7fa13
  (LGPL-3.0, 소스 및 수정 내역 보관). 원 논문의 원저자 배포판과 구분되는
  Rohwedder 등의 공개 재구현입니다.
- flx 논문: *Benchmarking fixed-length Fingerprint Representations across different Embedding Sizes and Sensor Types*, BIOSIG 2023.

모델 파일의 정확한 다운로드 주소와 해시는 각 `SOURCES.json`을 참고합니다.
