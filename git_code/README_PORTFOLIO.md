# KOSPI200 퀀트 투자 파이프라인

> **KOSPI200 주식을 대상으로 한 투트랙(Two-Track) 퀀트 투자 전략 시스템**  
> 본 프로젝트는 **포트폴리오 목적**으로 작성되었으며, 투자 자문이 아닙니다.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📸 데모

> **스크린샷 자리**: 백테스트 결과 차트, 랭킹 대시보드 등을 추가할 수 있습니다.

## 🚀 재현 방법

### 옵션 1: 분리 구조 사용 (권장) ⭐

**구조**: `data_drive/` (구글 드라이브) + `git_code/` (GitHub, 현재 폴더)

```bash
# 1. GitHub에서 git_code 클론 (이미 완료)
# 2. 구글 드라이브에서 data_drive 다운로드
#    링크: https://drive.google.com/drive/folders/1C-CfAwSpnfJawC6rCejj5QBTMqfDRjiS?usp=drive_link
#    test-main/data_drive/ 위치에 배치

# 3. 의존성 설치
pip install -e .

# 4. 데이터 검증
python scripts/check_data_files.py

# 5. 랭킹 산정 + 백테스트
python scripts/run_ranking_and_backtest.py
```

**예상 소요 시간**: 약 10-20분 (데이터 다운로드 제외)

**상세 가이드**: [`docs/SPLIT_STRUCTURE_GUIDE.md`](docs/SPLIT_STRUCTURE_GUIDE.md)

### 옵션 2: 자동 데이터 다운로드 (전체 파이프라인)

```bash
pip install -e .
python -m src.cli data-download
python -m src.cli track-a && python -m src.cli track-b --strategy bt120_long
```

**예상 소요 시간**: 약 30-60분 (인터넷 속도에 따라 다름)

### 옵션 3: 외부 드라이브 데이터 사용 (랭킹/백테스트만)

**사전 준비**: 외부 드라이브(Google Drive, OneDrive 등)에서 데이터 다운로드

```bash
# 1. 데이터 검증
python scripts/check_data_files.py

# 2. 랭킹 산정 + 백테스트 (원클릭)
python scripts/run_ranking_and_backtest.py
```

**또는 단계별 실행**:
```bash
# 2. 랭킹 산정
python -m src.pipeline.track_a_pipeline

# 3. 백테스트
python -m src.pipeline.track_b_pipeline bt120_long
```

**예상 소요 시간**: 약 10-20분 (데이터 다운로드 제외)

**상세 가이드**: [`docs/DATA_EXTERNAL_DRIVE_GUIDE.md`](docs/DATA_EXTERNAL_DRIVE_GUIDE.md) 참고

**예상 산출물 위치**:
- 랭킹: `data/cal_data/ranking_*.parquet`
- 백테스트 결과: `data/cal_data/bt_metrics_*.parquet`
- 실행 메타데이터: `artifacts/runs/<run_id>/manifest.json`, `metrics.json`

## 📊 결과 요약

### Holdout 구간 성과 (2023-01-31 ~ 2024-11-18)

| 전략 | Sharpe | CAGR | MDD | Calmar | Hit Ratio | 리밸런싱 수 |
|------|--------|------|-----|--------|-----------|------------|
| **bt120_long** ⭐ | **0.5689** | **6.86%** | **-10.27%** | **0.6679** | **60.87%** | 23 |
| bt120_ens | 0.4234 | 4.59% | -9.26% | 0.4954 | 65.22% | 23 |
| bt20_ens | 0.2431 | 3.04% | -18.39% | 0.1650 | 47.83% | 23 |
| bt20_short | -0.1826 | -4.52% | -15.56% | -0.2904 | 56.52% | 23 |

**주요 특징**:
- ✅ Track A 앙상블 적용: 4개 모델의 강점 결합으로 안정성 확보
- ✅ 과적합 방지: IC Diff 92%+ 감소로 일반화 성능 향상
- ✅ 거래비용 반영: cost_bps=10.0 적용 (Net 지표)
- ⚠️ **주의**: 일부 전략(bt20_short, bt20_ens)은 Holdout 구간에서 음수 수익률

### Dev 구간 성과 (2016-01-04 ~ 2022-12-29)

| 전략 | Sharpe | CAGR | MDD | Calmar | Hit Ratio | 리밸런싱 수 |
|------|--------|------|-----|--------|-----------|------------|
| bt120_long | 0.3136 | 4.78% | -21.97% | 0.2177 | 50.57% | 87 |
| bt120_ens | 0.0149 | -0.37% | -22.78% | -0.0164 | 51.72% | 87 |
| bt20_short | -0.0486 | -2.02% | -36.50% | -0.0554 | 48.28% | 87 |
| bt20_ens | -0.2570 | -4.64% | -35.91% | -0.1293 | 45.98% | 87 |

**주요 특징**:
- ⚠️ Dev 구간에서 대부분 전략이 낮은 성과 (시장 환경 영향)
- bt120_long이 가장 안정적인 성과 (Sharpe 0.31, CAGR 4.78%)

## 📚 구조 및 문서

### 핵심 문서

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: 파이프라인 단계(L0~Lx)와 데이터 흐름
- **[STRATEGY_CARD.md](docs/STRATEGY_CARD.md)**: 유니버스/리밸런싱/롱숏 규칙/비용/벤치마크
- **[DATA_CARD.md](docs/DATA_CARD.md)**: 데이터 출처/기간/빈도/결측치 처리/생존편향
- **[MODEL_CARD.md](docs/MODEL_CARD.md)**: 단기/장기 모델 정의, 학습/검증 분리, 한계

### 프로젝트 구조

```
test-main/
├── configs/              # 설정 파일들 (74개 YAML)
├── src/                  # 핵심 소스 코드
│   ├── data_collection/  # 데이터 수집 (L0-L4)
│   ├── pipeline/         # 파이프라인 오케스트레이션
│   ├── tracks/           # Track A/B 구현
│   └── utils/            # 유틸리티 (재현성 포함)
├── scripts/              # 실행 스크립트
├── data/                 # 샘플 데이터 (재현용)
├── artifacts/            # 실행 아티팩트 (runs/<run_id>/)
├── tests/                # 테스트 코드
└── docs/                 # 문서
```

## 🎯 프로젝트 개요

### Track A: 랭킹 엔진
- **목적**: 피처 기반 KOSPI200 종목 랭킹 생성
- **단계**: L8 (단기/장기 랭킹), L11 (UI Payload)
- **산출물**: `ranking_short_daily`, `ranking_long_daily`

### Track B: 투자 모델
- **목적**: 랭킹 기반 백테스트 전략 예시 제공
- **단계**: L6R (랭킹→스코어 변환), L7 (백테스트)
- **전략**: BT20 (20일 보유), BT120 (120일 보유)

## 🔧 설치 및 설정

### 1. 의존성 설치

```bash
pip install -e .
```

### 2. 설정 파일 확인

`configs/config.yaml` 파일의 경로 설정을 확인하세요:

```yaml
paths:
  base_dir: <프로젝트 루트 경로>
```

또는 환경 변수 사용:

```bash
export BASE_DIR=/path/to/project
```

## 📖 상세 문서

더 자세한 내용은 다음 문서를 참고하세요:

- **[전체 README](README.md)**: 상세한 사용법, 설정, 성과 분석
- **[재현 가이드](docs/REPRODUCE.md)**: 재현성 확보 방법
- **[설정 레퍼런스](docs/설정_레퍼런스.md)**: 설정 파일 상세 설명

## ⚠️ 주의사항

- **투자 자문 아님**: 본 프로젝트는 포트폴리오 목적으로 작성되었으며, 실제 투자 결정에 사용하지 마세요.
- **데이터 제한**: 대용량 데이터는 GitHub에 포함되지 않습니다. 재현을 위해서는 데이터 다운로드가 필요합니다.
- **재현성**: `artifacts/runs/<run_id>/manifest.json`을 참고하여 동일한 환경에서 재현할 수 있습니다.

## 📄 라이선스

본 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

---

**작성일**: 2026-01-19  
**최종 업데이트**: 2026-01-19 (성과 지표 재산출)  
**버전**: v1.0  
**상태**: 포트폴리오 완료
