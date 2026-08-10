# Semiconductor Process Notes

이 문서는 SECOM처럼 센서명이 익명화된 데이터에서도 반도체 제조 문제를 어떻게 데이터 문제로 해석할지 정리한 노트입니다. 특정 센서가 실제 어떤 장비 tag인지 단정하지 않고, 공정 조건, 설비 상태, 계측 결과, 수율 리스크를 연결하는 관점에 초점을 둡니다.

## Photo

Photo 공정은 회로 패턴을 웨이퍼 위에 전사하는 단계입니다. 주요 관심사는 CD, overlay, focus/exposure condition, resist coating 상태, develop 균일도입니다.

데이터 관점에서는 노광 조건, chuck 상태, track 장비 상태, 온습도, alignment residual, CD 계측값이 수율과 연결될 수 있습니다. 모델이 특정 센서를 fail risk 후보로 올렸다면 실제 현장에서는 lot, reticle, exposure recipe, chamber/tool 이력과 함께 확인해야 합니다.

## Etch

Etch 공정은 필요한 패턴을 남기고 불필요한 막을 제거하는 단계입니다. plasma power, gas flow, pressure, endpoint, chamber condition이 결과에 큰 영향을 줍니다.

데이터 관점에서는 RF power, pressure, gas MFC, endpoint signal, chamber clean 이후 경과 시간 같은 변수들이 중요합니다. 불량 위험 score가 올라간 경우 over-etch, under-etch, residue, profile shift 가능성을 직접 단정하기보다 chamber 상태와 계측 결과를 함께 확인하는 것이 맞습니다.

## Diffusion / Ion Implant / Anneal

Diffusion 계열 공정은 열처리나 이온주입을 통해 웨이퍼의 전기적 특성을 형성하는 단계입니다. 온도, 시간, dose, energy, ramp profile, furnace 또는 implant 장비 상태가 중요합니다.

데이터 관점에서는 온도 안정성, recipe step duration, beam current, dose uniformity, anneal profile 같은 신호가 품질 리스크와 연결됩니다. 작은 drift가 바로 불량으로 보이지 않더라도 누적되면 electrical test나 yield loss로 이어질 수 있습니다.

## Thin Film

Thin Film 공정은 CVD, PVD, ALD 등으로 막을 형성하는 단계입니다. 막 두께, 균일도, 조성, stress, particle 수준이 주요 품질 항목입니다.

데이터 관점에서는 gas flow, chamber pressure, temperature, plasma power, deposition time, precursor 상태, pump/chiller 상태가 후보가 됩니다. sensor candidate가 반복적으로 중요하게 나오면 막 두께 계측, chamber matching, PM 이력과 연결해서 확인해야 합니다.

## CMP / Cleaning

CMP는 웨이퍼 표면을 평탄화하고, Cleaning은 particle과 residue를 제거하는 단계입니다. pad/slurry/pressure/rotation, chemical concentration, brush 상태, drying condition이 중요합니다.

데이터 관점에서는 압력, 회전수, slurry flow, motor current, chemical 농도, rinse/dry 조건, particle 계측값이 연결됩니다. false alarm이 많더라도 missed fail 비용이 크면 early warning 기준을 낮게 잡고 review queue로 보내는 운영이 가능할 수 있습니다.

## How This Connects to the Project

SECOM 데이터는 센서명이 익명화되어 있으므로 `sensor_064` 같은 후보를 특정 공정 원인으로 바로 번역할 수 없습니다. 그래서 이 프로젝트에서는 root cause naming이 아니라 sensor candidate prioritization으로 해석합니다.

실제 현장 데이터가 주어진다면 다음 매핑이 필요합니다.

| Model Output | Field Mapping Needed |
|---|---|
| High fail-risk score | lot/wafer, product, route, operation, tool/chamber |
| Top sensor candidate | FDC tag, recipe step, unit, control/spec limit |
| Alarm candidate | metrology, inspection, E-test, defect map |
| Repeated drift | PM history, chamber clean, part replacement, consumable age |
| Threshold policy | review capacity, missed-fail cost, false-alarm cost |

핵심은 AI 모델이 공정 엔지니어를 대체하는 것이 아니라, 어떤 lot이나 sensor를 먼저 볼지 우선순위를 주는 것입니다.
