# C1: 4비트 분위수 보조정보를 이용한 Thresholding

완료한 S14–S16 실험 중 **C1과 필요한 기준선·공격 평가만** 독립 실행합니다.
V5/V6, B0–B4, C0/C2 실행기나 Fusion 모델을 import하지 않습니다.
기존 C1의 수식·등록 계획·평가 지표·공격자 설정을 유지합니다.

## 구조

- 고정된 ArcFace/DeepPrint 512차원 임베딩 캐시를 읽습니다. 특징 추출·원본 변경은 없습니다.
- 등록 5장의 평균을 0 기준으로 이진화한 512비트를 유지합니다. 질의는 겹치지 않는 1장입니다.
- TRAIN 등록 평균에 좌표별 Gaussian CDF를, 별도 TRAIN 질의에 선형 관계와 Gaussian 잡음을 적합합니다.
- 사용자 등록 시 각 비트 구간 내 상대 분위수 W를 좌표당 4비트로 양자화합니다.
- 512개 코드를 실제 256바이트로 패킹합니다. 얼굴·지문 각각 저장하며 BCH 보조정보는 별도입니다.
- 공통 512×16 표에서 tau를 조회합니다. 양자화 구간 전체의 likelihood를 16점 수치 적분합니다.
  숨겨진 원래 W나 등록 평균을 검증에 사용하지 않습니다. tau 범위는 기본 ±0.02입니다.
- 두 모달리티는 독립 처리합니다. 신경망 threshold 학습이나 사용자별 재학습은 없습니다.

공통 엔진 `model.py`는 과거 체크포인트 호환을 위해 float32/8비트 표현도 이해하지만,
공개 실행기 `c1.py`는 **4비트 C1만** 생성·평가합니다.

## 준비와 실행

Python 3.10+, NumPy, PyTorch, 실제 BCH 평가용 C++17 `g++`가 필요합니다.
기존 `feature_extractors/run.sh`는 로컬 `feature_extractors/.venv`를 사용합니다.
다른 준비된 환경에서는 `bash feature_extractors/run.sh`를 `python`으로 바꿉니다.
패키지·데이터·가중치를 자동 설치하거나 다운로드하지 않습니다.

데이터·임베딩·과거 실행 결과는 Git에 포함하지 않습니다. 아래를 로컬에서 준비합니다.

1. 전체 MUCT 3,755장/276명, U.are.U 520장/65손가락/7명의 정규화된 512차원 임베딩 캐시.
   `face_embeddings.npz`, `fingerprint_embeddings.npz`에 float32 `embeddings`와 원본 파일명 `files`가 필요합니다.
2. 완료한 bootstrap/source 실행 폴더: `config.json`, `status.json`, `splits.json`, `virtual_users.csv`,
   `samples.csv`, `train_pairs.csv`, `validation_pairs.csv`, `test_pairs.csv`,
   `best_threshold.pt`, `final_model.pt`. 체크포인트는 이력 해시 검증에만 사용합니다.
   준비 순서는 저장소 루트 README의 환경·데이터 준비 절차를 따릅니다.

프로젝트 루트에서 실행합니다. 현재 PC의 기본 source-run은
`thresholding/results/train_20261029_162319_486882`입니다.

```bash
# 읽기 전용: 이력·데이터·분할 확인. 모델·결과 폴더를 생성하지 않음.
bash feature_extractors/run.sh -m scripts.s16_quantile_thresholding.c1 --check-only --device cpu

# C1, tau=0, 누출 양성 대조군 + 독립 공격자 + 실제 BCH
bash feature_extractors/run.sh -m scripts.s16_quantile_thresholding.c1 --device cuda

# 다른 PC에서는 로컬 입력 경로 지정
python -m scripts.s16_quantile_thresholding.c1 --device cuda \
  --source-run /path/to/completed_bootstrap --embedding-dir /path/to/embedding_cache

# 실제 생체 데이터가 없는 합성 검증
python -m unittest scripts.s16_quantile_thresholding.test_c1 -v
```

모집단 적합은 4회 등록 계획, 공격자 학습은 별도 4회, 검증/시험은 각 1회 계획을 사용합니다.
프로토콜 seed는 17이며 사람 단위 분할은 source-run과 일치해야 합니다.
공격자는 조건당 4종, 기본 50 epoch입니다. 신경망 학습은 이 공격자들에만 적용합니다.
단계별 진행률·경과 시간·ETA를 출력합니다. `--help`에서 옵션을 확인할 수 있습니다.

결과는 새로운 `thresholding/results/c1_<시간>/`에 저장하며 기존 결과를 덮어쓰지 않습니다.
`comparison.csv`는 본인 오류/반경 복구율/타인 수락률, `attacks.csv`는 공개정보 공격,
`report.json`은 상세 지표입니다. 코드·입력·체크포인트 해시와 실제 저장 페이로드도 기록합니다.
모든 모델/공격자 선택이 끝난 뒤 시험 평가를 시작합니다.

## 해석과 보안 범위

- 누출 양성 대조군은 등록 비트를 의도적으로 드러내 공격 평가가 작동하는지 확인합니다. 운영 후보가 아닙니다.
- 고정 공격 6종과 독립 학습 공격 4종을 사용합니다. 입력은 공개 tau, 저장 코드에서 유도한 W,
  설정에 따라 BCH syndrome입니다. 영상·임베딩·정답 비트·신원 ID는 공격 입력이 아닙니다.
- `joint_mlp/helper_sketch`는 syndrome을 MLP에 입력합니다. BCH 제약을 직접 푸는 선형대수 공격이나
  모든 결합 공격을 구현한 것은 아닙니다. **관측 공격 성공률 0%는 보안 증명이 아닙니다.**
- CDF 추정 오차, 좌표 상관, BCH와 결합한 노출이 남아 있습니다. 전체 512비트 zero leakage나
  특정 보안 강도를 주장하지 않습니다. 4비트 양자화 자체가 안전성을 보장하지 않습니다.
- BCH 결과는 등록 비트 복구이며 완전한 FE.Gen/Rep의 최종 키 복구 평가가 아닙니다.
- 얼굴/지문 대응은 가상입니다. 같은 사람의 모든 손가락은 같은 분할에 둡니다.
  현재 시험 지문은 한 사람뿐이므로 다른 사람 지문에 대한 FAR은 null입니다.
- 기존 시험 집합을 반복 사용한 탐색적 실험입니다. 적합/학습에 시험 데이터는 사용하지 않습니다.
- `bits_tau.npz` 등 정답·질의 비트 파일은 오프라인 분석 자료이며 공개 보조정보가 아닙니다.

## 파일과 출처

`model.py`: 수식·표·패킹, `protocol.py`: 등록 계획, `evaluation.py`: hard-bit/BCH 평가,
`diagnostics.py`: tau 단독 대조군, `sketch.py`: 기존 BCH와 일치하는 syndrome,
`attacks.py`: 독립 공격, `reporting.py`: 원실험과 동일한 평균 방식의 결과표,
`c1.py`: 독립 실행기, `test_c1.py`: 합성·의존성 검증입니다.
공통 평가 코드는 원래 S11–S14 구현에서 C1에 필요한 부분을 분리했습니다.

기존 이론을 고정 등록 비트와 유한 저장량에 적용했으며 새로운 원천 보안 알고리즘을 주장하지 않습니다.

- [de Groot et al., Quantization in zero leakage helper data schemes (2016)](https://doi.org/10.1186/s13634-016-0353-z)
- [Stanko et al., Optimized quantization in Zero Leakage Helper data systems (2017)](https://doi.org/10.1109/TIFS.2017.2697840)
