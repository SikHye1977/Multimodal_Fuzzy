# Multimodal biometric thresholding experiments

고정된 얼굴·지문 512차원 임베딩을 비트열로 변환하는 Thresholding 연구 코드입니다.
실험 비교 보고서의 S5~S9와 그 실행에 필요한 공통 코드, 테스트, 특징 추출 소스를 선별했습니다.

| 실험 | 비교 내용 | 실행 스크립트 |
|---|---|---|
| S5 | 모달리티별 512차원 tau: zero / fixed / dynamic | `scripts/train_thresholding_512.py` |
| S6 | soft bit / STE / BCH 정정 범위 손실 A·B·C | `scripts/train_thresholding_bch_abc.py` |
| S7 | C 손실 고정, fixed / dynamic tau 비교 | `scripts/train_thresholding_c_fixed_dynamic.py` |
| S8 | V3 공통·모달리티별 scalar·512차원 tau | `scripts/train_thresholding_v3.py` |
| S9 | 등록 1장 기준선 / 등록 5장 평균 / 집합 통계 tau 학습, 질의 1장 | `scripts/train_thresholding_multisample_abc.py` |

## 포함 범위

`PUBLICATION_FILES.txt`가 커밋 대상으로 선정한 파일 목록입니다. 루트 `.gitignore`는
이 목록의 파일만 허용하므로 새 코드를 추가할 때 두 파일도 함께 갱신해야 합니다.

- 모델: `Thresholding_Model_512.py`, `Thresholding Model V3.py`, `Thresholding_Model_Set.py`.
- 이전 `train_*.py` 일부와 최초 `Thresholding Model.py`는 공통 데이터 처리, 손실,
  평가, 원실험 검증에 사용되므로 포함합니다. import만으로 학습을 시작하지 않습니다.
- `scripts/bch_evaluator.py`는 BCH 평가를 독립적으로 제공합니다. Fusion 모델과
  `scripts/train_fusion.py`는 포함하거나 import하지 않습니다.
- `scripts/fusion_bch_bridge.cpp`는 기존 이름을 유지한 C++ 연결 코드입니다.
  Fusion 신경망 의존성은 없습니다. BCH 후보와 등록 비트열의 완전 일치를 검사합니다.
- 얼굴·지문 특징 추출 코드는 포함하고, 외부 코드의 출처와 라이선스를 함께 보관합니다.

데이터셋, 가중치, 체크포인트, 임베딩, 개별 비트열, 실험 결과, 보고서의 원본 이미지와
가상환경은 포함하지 않습니다. ArcFace/DeepPrint 가중치는 각 `SOURCES.json`과
상위 배포처의 이용 조건에 따라 별도로 준비해야 합니다. `SOURCES.json`은 원실험의
출처·파일 해시 기록이므로 이 선별 저장소에 없는 원본 파일을 설명하는 항목도 있습니다.

## 환경과 데이터 준비

Python 3.10 이상, PyTorch와 NumPy를 사용합니다. 이미지에서 임베딩을 추출할 경우
`feature_extractors/requirements.txt`의 추가 의존성이 필요하며, 실제 BCH 복구 평가에는
C++17을 지원하는 `g++`가 필요합니다. 설치와 가중치 준비는 자동 실행하지 않습니다.

실험에 사용한 패키지와 실행 예시는 [특징 추출 설명서](feature_extractors/README.md)를
참고합니다. `feature_extractors/run.sh`는 `feature_extractors/.venv/bin/python`을 사용하므로
새 환경에서는 해당 가상환경을 준비하거나, 준비한 Python으로 스크립트를 직접 실행합니다.

MUCT와 U.are.U는 별도로 준비합니다. 두 데이터셋 사이에 실제 동일인 대응은 없으며,
이 코드의 얼굴·지문 대응은 가상 사용자 대응입니다. 얼굴은 사람 단위로, 지문은 한 사람의
모든 손가락을 묶어서 분할합니다. 사용자 ID는 분할·손실 라벨·평가용이며 모델 입력이 아닙니다.

전체 실행에는 다음 준비 순서가 필요합니다.

1. 데이터셋과 특징 추출기 가중치를 별도로 준비합니다.
2. `python -m feature_extractors.smoke_test --device cuda`로 임베딩과 manifest를 생성합니다.
   데이터 위치 등의 옵션은 `--help`로 확인합니다.
3. `scripts/train_thresholding.py`의 최초 실험을 완료해 기준 실행 폴더를 만듭니다.
   입력 경로와 학습 설정은 이 스크립트의 `--help`를 확인합니다.
4. S5~S9에 `--source-run <완료된_기준_실험_폴더>`와
   `--embedding-dir <임베딩_폴더>`를 지정합니다.

기존 기본 경로는 원실험의 로컬 경로입니다. 공개 저장소에는 그 결과가 없으므로
새 환경에서는 위 경로를 명시해야 합니다. `prepare_source()`는 설정·완료 상태·분할·쌍 목록,
`best_threshold.pt`, `final_model.pt` 및 코드/임베딩 해시를 검사합니다.
이 검사를 유지했으므로 코드만 받은 상태에서 곧바로 학습할 수는 없습니다.
다른 데이터나 설정으로 새 기준 실행을 만들면 기존 보고서와 동일한 결과를 재현하는 것은 아닙니다.

실험별 옵션과 지표 정의:

- [S5](scripts/thresholding_512.md)
- [S6](scripts/README_thresholding_bch_abc.md)
- [S7](scripts/README_thresholding_c_fixed_dynamic.md)
- [S8](scripts/README_thresholding_v3.md)
- [S9](scripts/README_thresholding_multisample_abc.md)

## 검증

프로젝트 루트에서 준비된 Python으로 실행합니다. 다음 테스트는 합성 입력과 임시 폴더를
사용하며 실제 생체 데이터셋을 학습하지 않습니다. BCH 테스트에는 `g++`가 필요합니다.

```bash
python -m unittest \
  scripts.tests.test_thresholding_512 \
  scripts.tests.test_thresholding_bch_abc \
  scripts.tests.test_thresholding_c_fixed_dynamic \
  scripts.tests.test_thresholding_v3 \
  scripts.tests.test_thresholding_multisample_abc -v
```

BCH 코드 자체의 테스트는 `make -C BCH test`로 실행할 수 있습니다.
`test_thresholding_bch_abc`에는 다섯 스크립트가 Fusion 모듈을 가져오지 않는지 확인하는 검사와
0~32비트 오류의 실제 복구 검사가 포함됩니다. 정정 범위 이내 비율, 실제 BCH 비트 복원율,
암호 키 복원율은 서로 다른 지표입니다. 이 저장소는 전체 FE의 키 생성·복구 보안을 검증하지 않습니다.
