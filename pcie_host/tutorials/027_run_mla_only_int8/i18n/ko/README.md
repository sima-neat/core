# 027 INT8 텐서로 MLA만 실행하기

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15분 |
| Model | any archive compiled for direct MLA input and output |
| Labels | PCIe, MLA, INT8, quantization, tensor |

## Concept

기본 PCIe 경로는 FP32 텐서를 카드로 보내고, EV74가 이를 양자화하고, MLA가 실행된 뒤, EV74가 결과를 다시 FP32로 역양자화합니다. 이미 INT8 데이터를 보유하고 있거나 양자화 단계를 직접 제어하려는 애플리케이션은 `ModelOptions.mla_only`를 설정할 수 있습니다. 그러면 카드는 MLA만 실행합니다. 호스트는 MLA 진입 계약에 맞는 INT8 텐서를 제출하고 원시 INT8 헤드를 받습니다. `model.info()`는 모든 텐서의 양자화 매개변수를 공개하므로 호스트는 하나의 수식으로 양자화와 역양자화를 수행할 수 있습니다.

```text
x = (q - zero_point) * scale
q = clamp(round(x / scale) + zero_point, -128, 127)
```

## Walkthrough

하나의 프로그램이 호스트에서 이미지를 양자화하고, MLA 전용 경로를 실행하고, 헤드를 역양자화한 뒤, 같은 큐의 기본 경로와 비교 검증합니다.

### MLA 전용 계약 확인하기 {#step-inspect-contract}

`mla_only`를 활성화하여 `Model`을 생성합니다. 이제 `info()`는 INT8 입력과 출력을 보고하며, 각각 `quant.scales[0]`과 `quant.zero_points[0]`을 가집니다. 입력에는 모델이 보정된 부동소수점 범위인 `input_range`도 포함됩니다. 입력이 여러 개인 모델은 제출 순서대로 입력마다 하나의 INT8 텐서를 나열합니다.

### 호스트에서 양자화하기 {#step-quantize-on-host}

이미지를 진입 형상에 맞게 크기를 조정하고, BGR을 RGB로 변환하고, 픽셀을 `input_range`에 매핑한 뒤, 진입 매개변수로 양자화 수식을 적용합니다. 같은 코드를 역양자화한 값을 보관해 두세요. 이 값이 기본 경로가 동일 조건 비교에 필요로 하는 정확한 FP32 입력입니다.

### INT8 경로 실행하기 {#step-run-int8}

`build()`는 MLA만 포함하는 카드 파이프라인을 시작합니다. `run()`은 INT8 텐서를 받아 `info().outputs`가 보고한 순서와 이름으로 출력마다 하나의 밀집 INT8 텐서를 반환합니다. 이 경로는 다른 dtype을 거부합니다. FP32 푸시는 카드에서 양자화되는 대신 실패합니다.

### 역양자화하고 비교하기 {#step-dequantize-and-compare}

`mla_only` 없이 두 번째 `Model`을 생성하고 역양자화된 FP32 값을 기본 경로로 보냅니다. 각 출력의 매개변수로 INT8 헤드를 역양자화하고 헤드별 최대 편차를 해당 헤드의 스케일 단위로 출력합니다. 두 경로 모두 같은 코드로 같은 MLA 프로그램을 실행하므로 오차는 0입니다.

## Run

[튜토리얼 설정](/tutorials/before-you-run)에 설명된 대로 PCIe 호스트 패키지를 설치하고 튜토리얼 번들을 다운로드합니다.

이 튜토리얼에는 MLA 입력과 출력을 직접 처리하도록 컴파일된 아카이브가 필요합니다. Model SDK의 `tessellate_parameters`에서 `enable_mla=True`를 지정하고, 모든 입력에 `HWC` DRAM 레이아웃을, 모든 출력에 `HWC16`을 지정합니다. Model Zoo 아카이브는 대신 EV74에서 테셀레이션을 수행하므로 `mla_only`가 활성화되면 거부됩니다.

```text
mla_only does not support stage 'tessellate_quantize_0_MLA_0/...' (tess)
```

조건을 충족하는 아카이브를 추출된 PCIe extras 루트에 예를 들어 `model_mlatess_int8.tar.gz`로 복사하고 그 경로를 `--model`로 전달합니다.

**Python:**

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/027_run_mla_only_int8/run_mla_only_int8.py \
  --model model_mlatess_int8.tar.gz
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_027_run_mla_only_int8 \
  --model model_mlatess_int8.tar.gz
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_027_run_mla_only_int8
./build/tutorials-standalone/tutorial_027_run_mla_only_int8 \
  --model model_mlatess_int8.tar.gz
```

MLA 입출력을 직접 처리하도록 컴파일된 YOLOv8n 아카이브를 사용하면 두 버전 모두 계약과 모든 헤드에 대한 0의 편차를 출력합니다.

```text
MLA-only contract:
  input images INT8 [640, 640, 3] scale=0.00391965 zero_point=-128 range=[0, 1]
  output bbox_0 INT8 [80, 80, 64] scale=0.0828159 zero_point=-60
  ...
Dequantized MLA-only outputs vs the default route (error in scale units):
  bbox_0 [80, 80, 64] max_err=0.0000
  ...
[OK] 027_run_mla_only_int8
```

기본값은 카드 0과 큐 0입니다. 다른 카드를 사용할 때만 `--card N`을 전달하세요.

## In Practice

애플리케이션이 양자화를 소유할 때 `mla_only`를 활성화하세요. 센서나 앞단 모델에서 이미 INT8을 생성하는 경우, 자체 후처리를 위해 원시 INT8 헤드가 필요한 경우, 또는 카드 측 지연 시간에서 EV74 단계를 제거하려는 경우입니다. 모든 스케일, 영점, 입력 범위는 `model.info()`에서 읽고, 모델의 다른 빌드에서 복사하지 마세요.

이 경로는 전부 아니면 전무입니다. 다중 입력 모델의 모든 입력은 INT8로 도착해야 하며, 이미지 전처리나 박스 디코드를 `mla_only`와 결합할 수 없습니다. 호스트가 FP32 데이터를 보유하고 양자화를 제어할 필요가 없다면 기본 경로를 유지하세요.

배포 진단은 [PCIe 모델 워크플로](/develop-apps/development-workflow/pcie-model/)에서 계속하세요.

## Source Files

- `run_mla_only_int8.cpp`
- `run_mla_only_int8.py`
- `../assets/street-scene.png`
