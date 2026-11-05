# 512차원 Thresholding 모델 실험

모델: `../Thresholding_Model_512.py`의 `ThresholdingModel512`.
실행: `scripts/s05_threshold512/train.py`. 기존 scalar 모델과 이전 결과는 수정하지 않는다.

## 구조

얼굴과 지문의 **정규화된 float32 512차원 임베딩**을 입력한다. 두 모달리티는 독립
파라미터를 사용한다. 등록과 질의는 각각 따로 모델에 입력한다. 상대 관측이나 사용자
ID는 모델 입력이 아니다. 출력은 다음과 같다.

| 출력 | 형태 | 의미 |
|---|---|---|
| `face_tau`, `fingerprint_tau` | `(B,512)` | 각 특징 좌표의 임계값 |
| `*_bits` | `(B,512)`, uint8 | `embedding_i >= tau_i`이면 1 |
| `*_soft_bits` | `(B,512)` | 학습용 `sigmoid((embedding_i-tau_i)/T)` |
| `*_distance` | `(B,512)` | 임계값과 해당 좌표의 절대 거리 |
| `*_reliability`, `*_reliability_logits` | `(B,512)` | 같은 신원의 다른 촬영과 비트가 일치할 확률 및 logit |
| `*_base_tau`, `*_delta_tau` | `(B,512)` | 공통 기준과 입력 의존 보정분 |

모드별 정의:

```text
zero:    tau(x) = 0
fixed:   tau(x) = 0.02 * tanh(theta),                  theta: 512차원
dynamic: tau(x) = 0.01 * tanh(theta) + 0.01 * tanh(g(x))
         g: Linear(512,32) → GELU → Linear(32,512)
```

`dynamic`이 입력별 512차원 임계값을 생성하는 주 실험이다. `fixed`는 데이터 전체에서
학습한 좌표별 임계값을 모든 입력에 동일하게 적용하는 비교군이다. 여기서 공통이라는
말은 사용자 간 공유를 뜻하며, 얼굴과 지문이 같은 임계값을 공유한다는 뜻은 아니다.
기존 특징 추출기는 고정하고, 특징을 별도 공간으로 투영하거나 비트 순서를 바꾸지 않는다.

모든 조건은 정확히 `tau=0`에서 시작한다. 신뢰도는 앞선 ABC 실험의 B 구조를
**두 모달리티와 세 조건에 동일하게 적용**하고, 같은 seed의 초기 가중치도 맞춘다.
신뢰도는 tau 단계가 끝난 뒤 따로 학습한다. 학습 전 모델의 신뢰도는 보정된 확률로
해석하면 안 된다. 선택된 신뢰도 모델이 학습된 가중치인지 `model.reliability_fitted`와
체크포인트의 `origin`으로 확인할 수 있다.

## 실행

기본 비교는 3조건 × 3 seed로 총 9회다.

```bash
bash /home/uxm2/hsh/2026_2/multimodal_biometric/feature_extractors/run.sh -u \
  -m scripts.s05_threshold512.train \
  --device cuda --modes zero fixed dynamic --seeds 17 29 43 \
  --threshold-epochs 100 --reliability-epochs 60 --patience 15 --batch-size 64
```

조건별 1회 비교는 `--seeds 17`, 입력 의존 모델만 실행하려면 `--modes dynamic`을
사용한다. GPU를 사용할 수 없는 환경에서는 `--device cpu`를 명시한다. 환경이나
패키지를 자동으로 수정하지 않고, GPU 실패 시 CPU로 조용히 전환하지 않는다.

진행률·현재 단계·epoch·loss·검증 결과·선택 epoch·경과 시간·ETA가 터미널에 표시된다.
ETA는 현재 단계의 속도와 남은 최대 epoch에 따른 추정이며 조기 종료와 최종 평가에
따라 달라진다. 자동 재개는 지원하지 않으며 기존 결과 폴더는 덮어쓰지 않는다.

입력만 확인하고 학습·출력 생성을 하지 않는 명령:

```bash
bash /home/uxm2/hsh/2026_2/multimodal_biometric/feature_extractors/run.sh \
  -m scripts.s05_threshold512.train \
  --check-only --device cpu
```

## 데이터와 loss

기존 전체 임베딩 `feature_extractors/results/full_dataset_check`를 사용하고 최초 실험
`thresholding/results/train_20261029_162319_486882`의 분할·가상 대응·평가 쌍·해시를
검증한다. 원본 이미지는 수정하지 않고 특징을 다시 추출하지 않는다.

- 학습/검증/시험 얼굴 사람 수: 220/28/28.
- 학습/검증/시험 지문 사람 수: 5/1/1. 같은 사람의 모든 손가락은 같은 분할에 속한다.
- 등록·질의 쌍 수: 2,985/347/367. 학습 촬영 조합은 epoch마다 바뀐다.
- 같은 seed·epoch에서는 조건 간 배치 순서를 동일하게 유지하고 해시로 검증한다.
- 검증·시험 분할을 통계 추정이나 가중치 업데이트에 사용하지 않는다.

tau loss는 기존 D 조건과 동일한 genuine soft-bit 거리, 다른 신원의 margin 거리,
좌표별 비트 균형, 샘플별 1 비율 제한으로 구성한다. 추가로 정규화한 등록·질의 tau 차이
제곱에 0.01, dynamic 보정분 크기 제곱에 0.001을 곱한다. soft bit의 온도 `T=0.02`는
원본 실험에서 가져온다. 새 quantization loss를 동시에 추가하지 않아 구조 변경의
영향을 비교한다.

tau 선택은 검증 hard bit 기준이다. tau=0 대비 타인과의 분리·동일인 오류·샘플별 균형이
허용 범위를 벗어나는 후보를 제외한다. 512차원 출력에서는 좌표별 상수 비트 비율과
좌표 균형 오차의 증가도 검사한다. 적격 후보 중 양쪽 오류가 `t=8` 이내인 비율,
모달리티 평균 `t` 이내 비율, 평균 Hamming 오류 순으로 선택한다. 초기 tau=0도
후보에 포함하고, 개선이 없으면 `fixed_zero_fallback`, best epoch 0으로 기록한다.

선택된 tau를 고정한 후 학습 촬영에서만 신뢰도용 변동 통계를 추정한다. 지문은
손가락 내 변동을 계산한 뒤 사람별 평균, 사람 간 평균 순으로 집계한다. 같은 genuine
등록·질의 비트의 일치 여부를 정답으로 하여 신뢰도 BCE를 학습한다. 신뢰도 loss는
tau에 역전파되지 않으며 매 epoch 임계값과 고정 통계가 바뀌지 않았는지 검사한다.

모든 조건의 모델 선택이 끝난 후 시험 분할을 평가한다. scalar oracle은 이번
512차원 모델의 최적 성능 한계를 나타내지 않으므로 새 보고서에 적용하지 않는다.

## 결과

새 결과 폴더는 `thresholding/results/threshold512_<날짜와시각>/`이다.

| 파일 | 내용 |
|---|---|
| `report.json`, `comparison.csv`, `aggregate.csv` | 전체 결과와 seed 평균/표준편차 |
| `paired_deltas.csv` | 같은 seed의 fixed−zero, dynamic−zero, dynamic−fixed 차이 |
| `<모드>_seed<번호>/best_model.pt` | 선택한 tau·신뢰도·고정 통계 |
| `<모드>_seed<번호>/best_threshold.pt` | fixed/dynamic 임계값 단계 선택 모델 |
| `<모드>_seed<번호>/history.jsonl` | loss·gradient norm·검증 결과·후보 제외 사유 |
| `<모드>_seed<번호>/*_outputs.npz` | **등록/질의 전체 512개 tau**, 비트·거리·신뢰도 logit |
| `<모드>_seed<번호>/*_tau_coordinates.json` | 512개 각 좌표의 tau 평균과 관측 간 표준편차 |
| `<모드>_seed<번호>/*_pairs_metrics.csv` | 쌍별 오류와 tau 평균/최솟값/최댓값/표준편차 |

tau 통계에서 `within_vector_std_mean`은 한 입력 안에서 좌표별 임계값이 얼마나 다른지,
`across_observations_std_mean`은 같은 좌표의 임계값이 입력에 따라 얼마나 달라지는지
나타낸다. fixed의 두 번째 값은 0에 가깝고 dynamic은 학습 결과에 따라 달라진다.

보고서의 `unique_pair_diagnostics`는 중복된 등록·질의 촬영 쌍을 제거한 오류와,
각 질의 내부의 비트 신뢰도 AUROC 평균·하위 10% 신뢰도 비트의 오류 비율도 제공한다.
전체 비트를 모은 AUROC와 혼동하지 않는다.

실제 지문 참여자 7명과 이미 확인한 시험 분할을 재사용하는 탐색 실험이다. 두 데이터셋의
연결은 실제 동일인 대응이 아니다. 새 이진화 모델은 새로운 등록에 적용하는 비교이며
이전에 등록한 비트·키의 호환성을 보장하지 않는다. 오류가 `t` 이내인 비율은 실제 BCH
실행이나 FE 키 복구율이 아니고, 이번 실험에 Fusion·`h_u` 생성은 포함하지 않는다.

## 모델 불러오기와 검사

```python
import torch
from Thresholding_Model_512 import ThresholdingModel512, Thresholding512Config

# 새 모델: 임계값 경로를 학습할 수 있는 상태로 생성된다.
model = ThresholdingModel512(Thresholding512Config(mode="dynamic"))
# out = model(face_embeddings, fingerprint_embeddings)
# out["face_tau"].shape == (batch_size, 512)

# 저장된 모델은 전용 로더를 사용한다. 기존 scalar 모델 로더와 호환되지 않는다.
# model, metadata = ThresholdingModel512.load_checkpoint("결과폴더/dynamic_seed17/best_model.pt", "cpu")
# with torch.inference_mode():
#     out = model(face_embeddings, fingerprint_embeddings)
```

합성 데이터만 사용하는 검증:

```bash
bash /home/uxm2/hsh/2026_2/multimodal_biometric/feature_extractors/run.sh \
  -m unittest scripts.tests.test_threshold512 -v
```
