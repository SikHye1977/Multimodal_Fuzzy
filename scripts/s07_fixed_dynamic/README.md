# C 손실 고정: fixed / dynamic 512차원 tau 비교

`scripts/s07_fixed_dynamic/train.py`는 기존 C 손실을 그대로 호출하면서 임계값 구조 두 가지를 비교한다.
기존 `Thresholding_Model_512.py`를 사용하며, 얼굴과 지문은 각각 별도 모델 파라미터를 갖는다.

| 구조 | 각 모달리티의 tau |
|---|---|
| fixed | 모든 입력에 공통인 학습 가능한 512차원 벡터: `0.02*tanh(theta)` |
| dynamic | `0.01*tanh(theta) + 0.01*tanh(MLP(x))`, MLP는 `512→32→512` |

두 구조 모두 처음에는 정확히 tau=0이다. 합계 범위는 ±0.02로 같지만 파라미터 수와 구조는 다르다.
dynamic은 등록과 질의 각각의 현재 임베딩만으로 tau를 예측한다. 상대편 이미지나 정답 신원은 모델에 입력하지 않는다.
전역 모델을 학습하며 사용자별로 추가 학습하지 않는다.

## 실행

```bash
cd /home/uxm2/hsh/2026_2/multimodal_biometric
bash feature_extractors/run.sh -m scripts.s07_fixed_dynamic.train \
  --device cuda --seeds 17 29 43 \
  --bch-t 32 --safety-margin 4 --epochs 100 --patience 15
```

**2개 구조 × 3개 시드 = 6회** 순차 실행한다. 학습률은 이전 C 실험과 같은 0.0001이다.
최대 100 epoch, 검증 선택 개선이 15 epoch 연속 없으면 해당 실험을 종료한다.
터미널에 진행률, 손실, 경과 시간, 단계·전체 예상 잔여 시간과 모달리티별 32비트 이내 도달률을 표시한다.

입력 확인만 할 때는 다음을 사용한다. 파일 생성, 환경 변경, 학습은 수행하지 않는다.

```bash
bash feature_extractors/run.sh -m scripts.s07_fixed_dynamic.train --check-only
```

## 손실과 성공 기준

- C 양성 쌍 손실: `mean((relu(Hamming - 28) / 32)**2)`.
- 순전파에는 실제 0/1 비트, 역전파에는 sigmoid STE를 사용한다.
- 기존 음성 쌍 분리, 좌표 균형, 비트열 내 1의 비율, tau 일관성, dynamic residual 제약도 유지한다.
- **성공은 BCH 적용 전 등록·질의 비트열의 Hamming 거리가 32 이하인 경우다.** 0비트 일치를 요구하지 않는다.
- 이번 전용 스크립트는 BCH 디코더를 빌드하거나 실행하지 않는다. 신뢰도 학습이나 FE 키 복구도 수행하지 않는다.

기존 참가자 분할과 가상 사용자 대응, 저장된 임베딩을 그대로 사용한다.
같은 시드의 fixed/dynamic은 동일한 학습 쌍과 배치 순서를 사용하며 해시로 확인한다.
원본 데이터와 기존 모델/실험 코드는 변경하지 않는다. 출발 모델을 이어 학습하지 않고 둘 다 tau=0에서 학습한다.

체크포인트 선택 시 기존의 비트 붕괴·타인 거리·FAR_h 악화 방지 조건을 유지한다.
조건을 통과한 후보 중 **얼굴과 지문 각각 중복 제거한 쌍의 32비트 이내 도달률을 1:1 평균**하여 우선 선택한다.
동률이면 평균 Hamming 거리가 최소 0.1비트 줄어든 모델을 선택한다. 두 모달리티 결과는 항상 별도로 함께 보고한다.
초기 tau=0도 선택 후보이므로 개선이 없으면 epoch 0이 선택된다.
모든 모델의 체크포인트 선택을 마친 뒤 시험 분할을 평가한다.

## 결과

새 폴더 `thresholding/results/c_fixed_dynamic_날짜_시간/`에 저장한다. 기존 결과는 덮어쓰지 않는다.

- `comparison.csv`: 구조·시드·분할·모달리티별 도달 건수/비율, 거리, FAR_h, tau 통계, tau=0 대비 변화.
- `aggregate.csv`: 시드 평균 및 표본 표준편차. 시드가 하나면 표준편차는 비워 둔다.
- `dynamic_minus_fixed.csv`: 같은 시드의 dynamic−fixed 차이. `within_radius_percent_delta`는 %p 단위이며 양수가 개선이다.
- `report.json`, `config.json`: 전체 결과, 설정, 입력·코드 해시, 평가 조건.
- `zero_validation.json`, `zero_test.json`: tau=0 기준선.
- 각 구조·시드 폴더의 `history.jsonl`, `training.log`: 모든 epoch의 손실·검증·선택/탈락 이유.
- `best_threshold.pt`, `last_threshold.pt`: 선택 및 마지막 체크포인트. 신뢰도 헤드는 미학습 상태다.
- `validation_bits_tau.npz`, `test_bits_tau.npz`: 등록·질의 512비트와 tau, Hamming 거리, 32비트 이내 여부.
- `validation_unique_pairs.csv`, `test_unique_pairs.csv`: 중복 제거한 개별 쌍의 결과.

dynamic이 실제로 입력에 따라 움직이는지는 `tau_across_unique_images_std_mean`과
`tau_genuine_mean_absolute_difference`로 확인한다. fixed에서는 둘 다 0이다.
`within_radius_fraction`은 0~1, `within_radius_percent`는 0~100으로 표기한다.

지문 검증·시험 분할은 각각 한 사람뿐이므로 다른 사람 지문 FAR_h는 null이다.
같은 사람의 다른 손가락 결과와 구분한다. 여러 시드는 참가자 수를 늘리지 않는다.
이미 관찰한 시험 분할을 재사용하므로 탐색적 비교다. 임계값이 바뀐 모델에는 새 등록이 필요하다.

합성 검증 명령(실제 데이터 학습 없음):

```bash
bash feature_extractors/run.sh -m unittest scripts.tests.test_fixed_dynamic -v
```
