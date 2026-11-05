# 512차원 Threshold: soft / hard STE / BCH 손실 비교

`train_thresholding_bch_abc.py`는 기존 `Thresholding_Model_512.py`와 저장된 얼굴·지문 임베딩을 사용한다.
특징 추출기, 원본 이미지, 기존 실험 결과 및 BCH 구현은 변경하지 않는다.

| 실험 | 학습 중 비트 표현 | 동일 신원 손실 |
|---|---|---|
| A | sigmoid soft bit | 기존 평균 절댓값 거리 |
| B | 순전파 0/1, 역전파 sigmoid STE | 평균 Hamming 거리 / 512 |
| C | B와 동일 | 평균 `[max(0, Hamming − (t − δ)) / t]²` |

기본값은 `t=32`, `δ=4`이며 C는 28비트 초과 오차에 양성 쌍 손실을 부여한다.
이는 실제 BCH 정정 범위 이내로 들어오도록 학습하는 목적함수이지 복구 보장은 아니다.
C는 B에 비해 손실의 형태와 기울기 크기가 함께 달라진다. 단순 손실값의 크기로 실험 간 성능을 비교하지 않는다.

타인 분리, 좌표별 비트 균형, 비트열 내부 1의 비율 제약, tau 안정성 및 dynamic residual 정규화는 모두 유지한다.
A는 이 항들도 soft bit로, B/C는 hard STE bit로 계산한다. 음성 쌍은 기존과 동일하게 신원별 균형과 가까운 다른 신원 4개를 이용한다.
지문 신원은 `(참가자, 손가락)`이며 사용자 ID는 손실용 라벨에만 쓰인다.

## 실행

```bash
cd /home/uxm2/hsh/2026_2/multimodal_biometric
bash feature_extractors/run.sh scripts/train_thresholding_bch_abc.py \
  --device cuda --arms A B C --modes fixed --seeds 17 29 43 \
  --bch-t 32 --safety-margin 4 --epochs 100 --patience 15
```

기본 실행은 **9회 순차 학습**이다. `fixed`는 모달리티마다 하나씩 학습하는 전역 512차원 tau로, 사용자 모두에게 공통으로 적용한다.
tau는 0에서 시작하고 기존과 같이 ±0.02로 제한한다. 얼굴과 지문 tau는 별도 파라미터다.
각 시드에서 A/B/C는 동일 초기 상태와 동일 등록·질의 쌍/배치 순서를 사용하고 해시로 확인한다.
시드별 학습 순서가 달라지며, 참가자 분할과 검증·시험 쌍은 고정된다.

입력에 따라 tau를 예측하는 기존 구조도 비교하려면 위 명령의 `--modes fixed`를
`--modes fixed dynamic`으로 바꾼다. 이 경우 **18회** 실행된다.
dynamic도 기존 설정(전역 tau ±0.01 + 입력별 residual ±0.01)을 사용한다.

학습 전 데이터와 실행 조건만 확인하려면:

```bash
bash feature_extractors/run.sh scripts/train_thresholding_bch_abc.py --check-only
```

이 명령은 학습·BCH 빌드·파일 생성을 하지 않는다. CUDA 사용이 불가능하면 실제 학습 명령은 오류로 종료하며,
CPU 실행을 원할 경우 명시적으로 `--device cpu`를 지정해야 한다.

기존 완료 실험 `thresholding/results/train_20261029_162319_486882`의 임베딩 해시, 참가자 분할, 가상 사용자 대응,
등록·질의 쌍을 검증한다. 참가자는 얼굴 220/28/28명, 지문 5/1/1명으로 분리되어 있다.
한 사람의 모든 손가락은 같은 분할에 속한다. 원본이나 특징 추출을 다시 처리하지 않는다.
`--source-run`은 동일 원본 실험 형식이어야 하며, 새 데이터셋을 임의로 연결하는 옵션이 아니다.

## 선택과 평가

매 epoch 실제 512비트로 검증하며, 결과 폴더에 빌드한 기존 BCH 구현으로 등록 비트의 정확한 복원을 검사한다.
Python 평가기는 `scripts/bch_evaluator.py`에 분리되어 있으며 Fusion Encoder/Decoder나
`train_fusion.py`를 불러오지 않는다. `fusion_bch_bridge.cpp`는 기존 이름을 유지한 C++ 연결 코드로,
Fusion 모델과 무관하다. 실행에는 C++17을 지원하는 `g++`가 필요하다.
검증의 타인 거리·AUROC·비트 균형 악화 방지 조건과 `FAR_h` 증가 제한을 통과한 후보 중 다음 순서로 선택한다.

1. 모달리티별 중복 제거한 등록·질의 쌍의 실제 BCH 정확 복원율 평균.
2. 동률이면 32비트 이내 비율 평균.
3. 다시 동률이면 동일 신원 평균 Hamming 거리 감소(기본 최소 0.1비트).

초기 tau=0도 선택 후보다. 개선이 없으면 `best_epoch=0`, `origin=fixed_zero_fallback`으로 남기며,
마지막 epoch의 모델과 손실·검증 기록도 별도로 저장한다. 기본 15 epoch 연속 선택 개선이 없으면 해당 학습을 종료한다.
모든 실험의 체크포인트 선택이 끝난 후 시험 분할을 평가한다.

- `FAR_h`: 다른 신원 쌍 중 Hamming 거리가 t 이하인 비율. 실제 FE 타인 수락률이 아니다.
- 얼굴은 다른 사람을, 지문은 다른 사람과 같은 사람의 다른 손가락을 구분해 보고한다.
- 검증·시험 지문에는 각각 한 사람만 있으므로 **다른 사람 지문 FAR_h는 null(측정 불가)**이다.
- 동일 지문 쌍이 가상 사용자 대응으로 반복되더라도 주 평가에서 중복 제거한다.
- BCH가 후보를 반환해도 등록 비트와 다르면 실패다. `bch_outcome`은 0=후보 없음, 1=정확 복원, 2=다른 후보다.
- 암호 키 Gen/Rep 검증과 신뢰도 학습은 포함하지 않는다. 체크포인트는 `threshold_only`, `reliability_fitted=false`다.
- 임계값 변경 후에는 새 등록이 필요하다. 기존 등록 키를 유지하는 공동 보정 실험과 구분한다.
- 이미 관찰한 시험 분할을 재사용하므로 탐색적 비교다. 여러 시드는 독립 참가자 수를 늘리지 않는다.

## 결과 파일

자동 생성되는 `thresholding/results/bch_abc_날짜_시간/`에 저장된다. 기존 디렉터리는 덮어쓰지 않는다.

- `comparison.csv`: 방식·시드·분할·모달리티별 거리/복원율/FAR_h, 선택 epoch, tau=0 대비 차이.
- `aggregate.csv`: 시드 평균과 표본 표준편차. 시드가 하나면 표준편차는 비워 둔다.
- `paired_deltas.csv`: 같은 시드의 B−A, C−B, C−A 차이.
- `report.json`, `config.json`: 전체 평가, 설정, 코드/입력 해시와 해석상 한계.
- 각 실험 폴더 `history.jsonl`, `training.log`: epoch별 손실, 검증 수치, 선택 및 탈락 이유.
- `best_threshold.pt`, `last_threshold.pt`: 선택된 모델과 마지막 모델.
- `validation_bits_tau.npz`, `test_bits_tau.npz`: 실제 등록/질의 512비트와 512차원 tau, 원본 쌍 인덱스, BCH 결과.
- `validation_unique_pairs.csv`, `test_unique_pairs.csv`: 중복 제거한 개별 쌍의 Hamming 거리와 BCH 결과.
- `zero_validation.json`, `zero_test.json`: tau=0 기준선.

터미널에는 epoch, 진행률, 손실, 경과 시간, 단계·전체 예상 잔여 시간이 표시된다.
ETA는 관측한 학습·검증 속도와 남은 최대 epoch를 사용하므로 조기 종료와 최종 평가에 따라 달라진다.

## STE 구현과 검증

`b_ST = b_hard + (p − stop_gradient(p))`를 사용한다.
순전파 값은 실제 비트와 같고, 역전파만 sigmoid의 기울기를 사용한다.
Hamming 거리는 `sum(b_e + b_q − 2*b_e*b_q)`로 계산한다.
정수 XOR 연산은 손실에 넣지 않는다. STE는 이산 함수의 정확한 미분이 아닌 대리 기울기다.

합성 데이터 테스트(실제 생체 데이터 학습 없음):

```bash
bash feature_extractors/run.sh -m unittest scripts.tests.test_thresholding_bch_abc -v
```
