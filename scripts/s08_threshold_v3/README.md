# Thresholding V3 실험

고정된 얼굴·지문 512차원 임베딩을 함께 받아 입력별 tau를 예측한다.
등록과 질의는 같은 모델을 **각각 별도로** 통과한다.
사용자 ID, 등록/질의 쌍 전체, FE 보조정보, 암호 키는 추론 입력이 아니다.

- 모델: 프로젝트 루트의 **Thresholding Model V3.py**
- 실험: **scripts/s08_threshold_v3/train.py**

## 구조와 네 가지 비교

| 모드 | 출력 | 이진화 |
|---|---|---|
| shared_scalar | 공통 tau (B,1) | 같은 scalar를 얼굴·지문 모든 좌표에 적용 |
| separate_scalar | 얼굴/지문 tau 각각 (B,1) | 모달리티별 scalar 적용 |
| shared_512 | 공통 tau (B,512) | 같은 벡터를 얼굴·지문 좌표별로 적용 |
| separate_512 | 얼굴/지문 tau 각각 (B,512) | 모달리티별 벡터 적용 |

최종 얼굴·지문 비트열 길이는 각각 512다.
공통 tau는 두 모달리티가 같은 비트열을 가져야 한다는 뜻이 아니다.

~~~mermaid
flowchart LR
  F["얼굴 임베딩 512"] --> FB["얼굴 branch 512→128→64"]
  P["지문 임베딩 512"] --> PB["지문 branch 512→128→64"]
  FB --> J["결합 128→128→64"]
  PB --> J
  J --> H["공통 head 또는 모달리티별 head"]
  FB -. "분리 head의 추가 입력" .-> H
  PB -. "분리 head의 추가 입력" .-> H
  H --> T["tau: 1 또는 512차원"]
  T --> B["각 원본 임베딩과 비교: x ≥ tau"]
  F --> B
  P --> B
  B --> O["얼굴 512 bits / 지문 512 bits"]
~~~

각 branch와 결합부는 Linear–LayerNorm–GELU를 두 번 사용한다.
공통 head는 결합 특징 64차원, 분리 head는 해당 branch 64차원과 결합 특징 64차원을 받는다.
Head는 Linear–LayerNorm–GELU–Linear를 사용한다. BatchNorm과 dropout은 없다.

- tau = a·tanh(base_logit) + b·tanh(head(input)).
- 기본값 a=b=0.01, 전체 범위 [-0.02,0.02].
- base는 모든 사용자가 공유하는 학습 파라미터다.
- base와 마지막 Linear를 0으로 초기화하여 네 모드 모두 정확한 tau=0 비트열에서 시작한다.
- branch는 tau 예측에만 사용한다. 비트열은 **원본 임베딩과 tau**를 비교한다.
- 기존 캐시의 L2 정규화 임베딩을 사용한다. 추출기 재학습·추가 projection은 하지 않는다.

분리 head에서도 다른 모달리티의 변화가 자기 tau에 영향을 줄 수 있다.
서로 독립적인 임베딩 좌표에 공통 tau를 적용하는 것이 유리한지는 네 모드 비교로 확인한다.

## 학습 데이터와 손실

기존 완료 실험의 임베딩 해시, 사람별 분할, 가상 사용자 대응, 고정 검증/시험 쌍을 검증하고 재사용한다.
학습 쌍은 기존 방식대로 epoch별로 구성하며 같은 seed/epoch의 배치는 모든 모드에서 동일하다.
얼굴은 사람 ID, 지문은 (사람 ID, 손가락 ID)가 동일인 쌍의 라벨이다.
ID는 손실과 평가에만 사용한다. 정답 tau나 정답 암호 키는 없다.

~~~text
등록 얼굴 + 등록 지문 → 모델 → 등록 얼굴 bits, 등록 지문 bits
질의 얼굴 + 질의 지문 → 모델 → 질의 얼굴 bits, 질의 지문 bits
                       ↓
             모달리티별 동일/상이 신원 거리 손실
~~~

순전파에는 hard 0/1 bits를 사용하고 역전파에는 sigmoid 기반 STE를 사용한다.

~~~text
soft = sigmoid((x-tau)/T)
STE = hard.detach() + soft - soft.detach()
~~~

기본 온도 T=0.02다. STE는 hard 비교의 **대리 기울기**이며 정확한 미분은 아니다.
모달리티별 손실을 계산한 뒤 얼굴·지문 평균을 사용한다.

1. **C 동일인 손실**: [max(Dg-28,0)/32]².
   기본 반경 t=32, 학습 여유 4비트다. 28비트 이하인 쌍에 추가 축소를 강요하지 않는다.
   같은 신원의 반복 수에 따른 편향을 줄이도록 배치 안에서 신원별 가중치를 맞춘다.
2. **상이 신원 손실**: [max(M-Di,0)/M]².
   얼굴 M=224, 지문 M=160을 시작값으로 사용한다.
   전체 다른 신원 평균과 가까운 상위 4개 다른 신원 손실을 0.5씩 섞는다.
   같은 신원의 반복 표본은 negative에서 제외한다.
3. **좌표별 비트 균형**: 신원 가중 평균 1 비율이 0.5에서 벗어나는 정도의 제곱 평균.
4. **개별 비트열 편중 방지**: 1 비율이 [0.3,0.7]을 벗어나면 벌점.
5. **입력별 tau 변화량 억제**: (delta_tau/b)² 평균.

가중치는 동일인 1, 상이 신원 1, 좌표 균형 1, 개별 편중 1, tau 변화량 0.001이다.
등록·질의 tau를 같게 만드는 손실은 넣지 않는다.
이 값들은 실험 시작 설정이며, 85비트에서 32비트로의 개선을 보장하지 않는다.
비트 균형은 비트 간 독립성이나 암호학적 엔트로피를 보장하지 않는다.

## 평가와 체크포인트 선택

- 실제 hard bits로 **동일인 해밍거리 ≤32 비율**을 측정한다. 32는 성공, 33은 실패다.
- 주 지표: 모달리티마다 각 신원의 성공률을 동일 가중 평균한 뒤, 두 모달리티 결과를 평균한다.
- 평균 거리, p95, 0비트 오류율, 두 모달리티 모두 ≤32인 비율도 저장한다.
- 상이 신원 FAR_h=P(Di≤32), 거리 평균/p05, AUROC를 저장한다.
  지문은 같은 사람의 다른 손가락과 다른 사람을 구분한다.
- **한 모달리티만 다른 신원으로 교체**한 입력도 검사한다.
  각 평가 쌍에서 같은 분할 안의 다른 신원 질의 하나를 결정적으로 선택한다.
  전체 가능한 교체 조합을 전수 평가하는 검사는 아니다.
- 다른 모달리티가 달라지면 같은 이미지 쌍도 tau가 달라질 수 있다.
  따라서 (등록 얼굴, 등록 지문, 질의 얼굴, 질의 지문)가 모두 같은 행만 중복 제거한다.
  단일 모달리티 이미지 쌍을 기준으로 하는 보조 지표는 **모든 동반 입력 조건을 평균**한다.

검증 분할에서 주 성공률이 가장 높은 모델을 선택한다.
동률이면 평균 해밍거리가 0.1비트 이상 감소한 모델을 선택한다.
기본 선택 조건:

- 두 모달리티 각각의 신원 평균 ≤32 비율이 tau=0보다 감소하지 않을 것.
- 이용 가능한 모든 상이 신원 범주와 단일 모달리티 교체의 FAR_h가 tau=0보다 증가하지 않을 것.
- AUROC 감소 ≤0.001.
- 좌표 균형 오차 증가 ≤0.01, 상수 좌표 비율 증가 ≤0.02, 편중 비트열 비율 증가 ≤0.01.

조건은 --max-* 옵션으로 조정할 수 있다.
상이 신원 **평균 거리 자체의 감소**는 탈락 조건으로 사용하지 않는다.
매 epoch 수치와 탈락 이유는 history.jsonl에 저장한다.
선택 epoch=0이면 학습을 실행하지 않았다는 뜻이 아니라 조건을 충족하는 개선 체크포인트가 없었다는 뜻이다.
마지막 학습 결과는 last_threshold.pt에 별도로 남긴다.

시험은 모든 체크포인트 선택 후 평가한다.
이미 시험 결과를 반복 확인한 연구이므로 새로운 독립 확증 시험으로 해석하지 않는다.
지문 분할은 학습/검증/시험 각각 실제 사람 5/1/1명이다.
검증/시험에서 다른 사람 지문 FAR_h는 산출할 수 없으며 null로 기록한다.
가상 얼굴·지문 대응, 반복 입력 조건, seed 반복이 실제 참여자 수를 늘려주지는 않는다.

## 실행

기존 환경을 사용한다. 의존성 설치나 환경 변경은 필요하지 않다.

기본 네 모드 × seed 17/29/43 = **12회 순차 실행**:

~~~bash
cd /home/uxm2/hsh/2026_2/multimodal_biometric
bash feature_extractors/run.sh -m scripts.s08_threshold_v3.train \
  --device cuda --seeds 17 29 43 --epochs 100 --patience 20
~~~

입력 파일만 검증하고 끝내기:

~~~bash
bash /home/uxm2/hsh/2026_2/multimodal_biometric/feature_extractors/run.sh \
  -m scripts.s08_threshold_v3.train --check-only
~~~

한 모드/한 seed만 실행:

~~~bash
bash /home/uxm2/hsh/2026_2/multimodal_biometric/feature_extractors/run.sh \
  -m scripts.s08_threshold_v3.train --device cuda --modes separate_512 --seeds 17
~~~

기존 독립 입력 512차원 모델을 같은 손실로 추가 비교하려면 모드를 명시한다(15회):

~~~bash
bash /home/uxm2/hsh/2026_2/multimodal_biometric/feature_extractors/run.sh \
  -m scripts.s08_threshold_v3.train --device cuda \
  --modes shared_scalar separate_scalar shared_512 separate_512 independent_512
~~~

independent_512는 기존 Thresholding_Model_512.py의 dynamic 구조를 새로 학습한다.
그 모델에 포함된 reliability head는 고정되고 호출하지 않는다.
기존 실험 체크포인트를 이어 학습하는 옵션은 아니다.

tau 폭: --tau-base-bound, --tau-delta-bound.
상이 신원 margin: --face-margin-bits, --fingerprint-margin-bits.
검증 결과에 따라 정하고 시험 결과에 맞춰 선택하지 않는다.
CUDA가 없으면 자동으로 CPU 학습을 시작하지 않는다. CPU는 --device cpu를 명시한다.
진행률과 현재 단계/전체 ETA가 터미널에 표시된다. ETA는 최대 epoch 기준 추정으로 조기 종료에 따라 달라진다.

## 저장 결과

기본 위치: thresholding/results/v3_<실행시각>/.

- config.json: 설정, 코드/입력 해시, 평가 정의.
- report.json: 전체 지표, tau=0 기준선, 비교 결과.
- comparison.csv: 모드·seed·분할·모달리티별 지표와 tau=0 대비 차이.
- aggregate.csv: seed별 평균/표준편차.
- shared_schedule_sha256.json: 동일 seed/epoch 배치 일치 기록.
- 모드별 best_threshold.pt, last_threshold.pt, history.jsonl, training.log.
- validation_bits_tau.npz, test_bits_tau.npz: 등록/질의 tau와 uint8 비트열, 거리, 반경 판정, 교체 donor 행.
- *_context_pairs.csv: 입력 파일명과 쌍별 거리.
- status.json: 실행/완료/실패 상태.

기존 폴더를 덮어쓰지 않으며 중단 후 이어 학습은 지원하지 않는다.
이 실험은 비트열 거리를 평가한다. 실제 BCH 복호·FE 키 복구 성공률을 계산하지 않는다.
새 모델을 배포하면 해당 모델로 등록 비트열과 FE 보조정보를 다시 만들어야 한다.
기존 등록 비트열·보조정보와 새 threshold 모델을 임의로 혼용하지 않는다.

## 모델 호출

파일명에 공백이 있으므로 importlib로 불러온다.

~~~python
import importlib.util
import sys
import torch

path = "/home/uxm2/hsh/2026_2/multimodal_biometric/Thresholding Model V3.py"
spec = importlib.util.spec_from_file_location("thresholding_model_v3", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

model, metadata = module.ThresholdingModelV3.load_checkpoint(
    "/absolute/path/to/best_threshold.pt", device="cuda"
)
# face_embedding, fingerprint_embedding: L2 정규화 float32, CUDA, (B,512)
with torch.no_grad():
    output = model(face_embedding, fingerprint_embedding)
face_tau = output["face_tau"]
fingerprint_tau = output["fingerprint_tau"]
face_bits = output["face_bits"]           # uint8, (B,512)
fingerprint_bits = output["fingerprint_bits"]
~~~

합성 데이터 검증:

~~~bash
bash /home/uxm2/hsh/2026_2/multimodal_biometric/feature_extractors/run.sh \
  -m unittest scripts.tests.test_threshold_v3 -v
~~~

테스트는 합성 데이터만 최적화하며 실제 생체 데이터를 학습하지 않는다.
