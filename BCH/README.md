# 512비트 BCH 신드롬 복구 모듈

`/home/uxm2/hsh/DID-Wallet/ios/CPP/BCH`의 신드롬 계산, Berlekamp-Massey,
Chien Search 흐름을 참고한 독립 C++17 구현입니다. 원본 프로젝트는 변경하지 않습니다.
외부 라이브러리, iOS 또는 React Native 의존성이 없습니다.

## 입력과 범위

- 입력: **이미 이진화된 512비트**, 정확히 64바이트로 포장합니다.
- ArcFace/DeepPrint의 실수 임베딩을 직접 받거나 임계값을 계산하지 않습니다.
  각 임베딩 좌표를 공통 scalar tau로 이진화한 후 전달해야 합니다.
- `GF(2^10)`, 원시다항식 `x^10 + x^3 + 1` (`0x409`).
- 모코드 길이 1023에서 좌표 0~511만 유지하고, 512~1022는 고정 0으로 처리합니다.
- 기본 정정 반경 `t=8`. 생성자에서 `1 <= t <= 255`를 지정할 수 있습니다.
  큰 t는 더 큰 보조정보를 요구하며, 키 엔트로피 보장은 이 모듈의 범위가 아닙니다.
- 512개 좌표 모두 복구 대상입니다. 255, 256, 511번 비트도 생략하지 않습니다.
- 비트 번호 0은 첫 바이트의 최상위 비트입니다. 비트 번호 i를 다항식 x^i의 계수로 봅니다.
- 임의의 생체 비트열과 그 신드롬을 다루는 secure-sketch 구성 요소입니다.
  512비트 메시지에 패리티를 붙이는 systematic encoder나 완성된 FE.Gen/FE.Rep는 아닙니다.

## 빌드와 합성 데이터 검사

```bash
cd /home/uxm2/hsh/2026_2/multimodal_biometric/BCH
make all
make test
make example
make sanitize
```

`make all`은 `build/libbch512.a`와 예제 실행 파일을 만듭니다.
컴파일러는 C++17 지원 `g++`를 사용하고 `make CXX=...`로 지정할 수도 있습니다.
`sanitize`는 AddressSanitizer와 UndefinedBehaviorSanitizer를 사용합니다.
검사는 합성 비트열만 사용하며 얼굴/지문 데이터에 접근하지 않습니다.

## 사용 예

```cpp
#include "BCH.hpp"

BCH bch(8);

// enrollment_bits와 query_bits는 각각 길이가 정확히 512인 "0"/"1" 문자열.
auto enrolled = BCH::from_bitstring(enrollment_bits);
auto saved_syndromes = bch.compute_syndrome(enrolled);
auto helper_bytes = bch.pack_syndromes(saved_syndromes);

auto query = BCH::from_bitstring(query_bits);
auto result = bch.recover(query, bch.unpack_syndromes(helper_bytes));

if (result.decoded()) {
    auto recovered_bits = BCH::to_bitstring(result.data);
    // 다음 FE 단계에서 등록 때와 같은 방식으로 키를 재생성하고 검증한다.
} else {
    // BCH 정정 실패. 현재 얼굴/지문과 공동 복호정보를 이용하는 보정 경로의 입력으로 사용 가능.
}
```

`decoded()`가 true여도 실제 FE의 키 검증에 실패하면 개별 복구 실패로 처리해야 합니다.
검증을 위해 실제 등록 원본이나 비밀키를 평문 저장하라는 의미는 아닙니다.

## 복구 결과의 정확한 의미

| 상태 | 의미 |
|---|---|
| `NoChange` | 입력과 저장된 신드롬이 일치하여 비트를 바꾸지 않았습니다. |
| `Corrected` | t개 이내 위치를 변경한 후보가 저장된 신드롬과 일치합니다. |
| `Uncorrectable` | 유효한 후보를 얻지 못했습니다. `data`와 `corrected_positions`는 비어 있습니다. |

정확한 등록 신드롬과 **t개 이하의 비트 오류**라는 조건에서 원래의 512비트를 복구합니다.
오류가 t개를 넘으면 실패하거나 다른 신드롬 일치 후보를 반환할 수 있습니다.
신드롬 재검사는 모든 과다 오류나 다른 등록 정보 사용을 검출하는 인증 절차가 아닙니다.
따라서 성공 상태 자체를 등록 키와의 완전 일치로 간주하면 안 됩니다.
테스트에는 다른 비트열이 같은 신드롬을 갖는 경우와 그쪽으로 정정되는 경우도 포함합니다.

입력 길이, 신드롬 개수/원소 범위, 문자열 형식 오류는 `std::invalid_argument`입니다.
이는 품질 저하에 따른 정상적인 `Uncorrectable` 결과와 구별합니다.
입력 데이터를 출력하는 디버그 로그는 없습니다.

## FE 보조정보 표현

`compute_syndrome()`은 기존처럼 `[0, S1, ..., S(2t)]`를 반환합니다.
원소는 0~1023의 정수이며, 첫 원소는 사용하지 않는 0입니다.

`pack_syndromes()`는 첫 원소를 제외하고 각각 **10비트**, 상위 비트 우선으로 포장합니다.
마지막 바이트에 남는 비트는 0이고, 역변환 시 0인지 검사합니다.

- 기본 `t=8`: 유효 신드롬 16개, **160비트 = 20바이트**.
- 일반 t: 유효 비트 `20*t`, 저장 바이트 `(20*t+7)/8`의 정수 나눗셈.
- 위 수치는 이 직렬화 방식의 저장량입니다. 신드롬 원소의 독립성이나 정보 누출량을 뜻하지 않습니다.
- 버전, t, 모달리티 식별자, 무결성 보호 등에 필요한 별도 메타데이터는 포함하지 않습니다.
  `BCH512-v1`, m=10, polynomial=0x409, input_bits=512, t, bit order를 호출 측에서 관리해야 합니다.
- 이 값은 BCH의 FE 보조정보이며, 학습된 공동 복호정보 `h_u`와 다릅니다.

이전 GF(2^8)의 신드롬이나 2자리 hex 원소 직렬화와는 호환되지 않습니다.
`recover()`의 반환형도 기존 바이트 벡터에서 `RecoveryResult`로 변경했습니다.

## 검사 범위

- 512개 모든 위치에 대한 단일 비트 오류 복구.
- 앞/뒤 경계를 가로지르는 오류와 임의 입력의 0~8개 오류 복구.
- 다른 t 설정과 허용 범위 양 끝의 검사.
- 로그 테이블을 사용하지 않는 독립 다항식 연산과 신드롬 교차 검증.
- 독립적으로 구성한 BCH 생성다항식/코드워드의 신드롬 검증.
- 길이/형식/범위 오류, 보조정보 직렬화, 단축 구간 밖 오류 위치 거부.
- 정정 반경 초과 입력 처리와 과다 오류 검출의 한계 확인.

생체 데이터 전처리, 모델 학습, 키 추출, 실제 생체 복구율 평가는 포함하지 않습니다.
