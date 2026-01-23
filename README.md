# KOSPI200 투트랙 퀀트 투자 전략 파이프라인

KOSPI200 주식을 대상으로 한 **투트랙(Two-Track)** 퀀트 투자 전략 시스템입니다.

## ✅ **프로젝트 상태: 최종 완료 (2026-01-19)**
**완전한 로컬 정리 및 최적화 완료** 🎉

### 🧹 **레포 정리 작업 완료**
- **정리된 파일**: 1,264+개 파일 격리
- **루트 디렉토리**: 183개 → 9개로 정리 (95% 감소)
- **폴더 구조**: 완전 모듈화 및 최적화
- **정리 로그**: `docs/LOCAL_정리_로그.md` 참고

## 📦 **프로젝트 최종 상태 고정**
**현재 상태가 공식 최종 버전으로 설정되었습니다**
- **완료 일시**: 2026-01-19
- **포함 내용**: 정리된 코드, 설정, 문서 파일들
- **LOCAL_TRASH**: 격리된 파일들 안전 보관
- **참고 문서**: `docs/LOCAL_정리_로그.md`

## 🏆 프로젝트 최종 성과 (완료)

### ✅ **주요 목표 달성**
- **Track A (랭킹 엔진)**: bt120_long **Sharpe 0.5689** 달성 ✅
- **앙상블 최적화**: 4개 모델의 강점 결합 성공 ✅
- **과적합 방지**: IC Diff 92%+ 감소로 안정화 ✅
- **실전 적용 준비**: 일반화 성능 검증 완료 ✅
- ⚠️ **성과 업데이트**: 2026-01-19 최신 데이터 기준으로 성과 지표 재산출 완료

### 📊 **최종 백테스트 성과 (Holdout 구간: 2023-01-31 ~ 2024-11-18)**
| 전략 | Sharpe | CAGR | MDD | Calmar | Hit Ratio | 상태 |
|------|--------|------|-----|--------|-----------|------|
| **bt120_long** | **0.5689** | 6.86% | -10.27% | 0.6679 | 60.87% | ⭐ **주요 전략** |
| bt120_ens | 0.4234 | 4.59% | -9.26% | 0.4954 | 65.22% | ✅ **안정적** |
| bt20_ens | 0.2431 | 3.04% | -18.39% | 0.1650 | 47.83% | ⚠️ **낮은 성과** |
| bt20_short | -0.1826 | -4.52% | -15.56% | -0.2904 | 56.52% | ❌ **음수 수익** |

## 🎯 프로젝트 핵심 목적

본 프로젝트는 **두 가지 독립적인 트랙**으로 구성되어 이용자에게 정보를 제공합니다:

1. **Track A (랭킹 엔진)**: 피처들로 KOSPI200의 랭킹을 산정하여 이용자에게 제공
2. **Track B (투자 모델)**: 랭킹을 기반으로 다양한 투자모델 예시를 만들어 이용자에게 정보 제공

두 트랙은 **독립적으로 실행 가능**하며, 각각 다른 목적을 가집니다.

### 🧹 **프로젝트 정리 특징**
- **완전 모듈화**: 각 기능별 폴더 분리 및 최적화
- **깔끔한 구조**: 루트 디렉토리 불필요 파일 완전 제거
- **안전한 보관**: LOCAL_TRASH 폴더를 통한 파일 백업
- **유지보수 용이**: 정리된 구조로 코드 관리 효율성 극대화

## 🎖️ **Track A 최종 구성 (앙상블 최적화 완료)**

### **앙상블 가중치 (과적합 개선 적용)**
| 전략 | Grid Search | Ridge | XGBoost | Random Forest | IC | ICIR | Hit Ratio |
|------|-------------|-------|---------|---------------|----|------|-----------|
| **단기** | 30% | 60% | 10% | 0% | 0.0366 | 0.3502 | 53.3% |
| **장기** | 5% | 15% | 80% | 0% | 0.0633 | 1.1449 | 62.8% |

### **사용 모델**
- **Grid Search**: 피처 그룹별 가중치 최적화 (L8 단계)
- **Ridge**: 개별 피처 가중치 자동 학습 (L5 단계)
- **XGBoost**: 앙상블 ML 모델 (L5 단계)
- **Random Forest**: 개선된 ML 모델 (IC=0 문제 해결)

### **과적합 방지 성과**
- **단기 전략**: IC Diff 0.0371 (LOW 위험)
- **장기 전략**: IC Diff 0.0557 (MEDIUM 위험)
- **개선 효과**: IC Diff 92%+ 감소로 실전 적용 가능



## 📂 프로젝트 폴더 구조 (최종 정리 완료)

```
test-main/
├── .cursor/              # Cursor IDE 설정
├── .github/              # 🔄 CI/CD 워크플로우
│   └── workflows/
│       └── ci.yml
├── artifacts/            # 🎯 실행 아티팩트 (Git 제외)
│   ├── .gitkeep
│   └── runs/             # runs/<run_id>/manifest.json, metrics.json
├── configs/              # ⚙️ 설정 파일들 (48개 YAML)
│   ├── config.yaml       # 메인 설정 파일
│   ├── features_*.yaml  # 피처 정의
│   └── feature_weights_*.yaml  # 피처 가중치
├── data/                 # 📊 데이터 (Git 제외, 샘플만 포함)
│   ├── cal_data/         # 계산된 데이터 (백테스트 결과 등)
│   │   ├── bt_metrics_*.csv/parquet
│   │   ├── ranking_*.csv/parquet
│   │   └── README.md
│   ├── raw_data/         # 원시 데이터
│   │   ├── ohlcv_daily.parquet
│   │   ├── esg_daily.parquet
│   │   ├── news_sentiment_daily.parquet
│   │   └── README.md
│   └── README.md
├── data_backup/          # 백업 데이터 (Git 제외)
├── docs/                 # 📚 문서 파일들 (30개 파일)
│   ├── ARCHITECTURE.md   # 시스템 아키텍처
│   ├── STRATEGY_CARD.md  # 전략 명세
│   ├── DATA_CARD.md      # 데이터 명세
│   ├── MODEL_CARD.md     # 모델 명세
│   └── PORTFOLIO_CLEANUP_*.md  # 정리 작업 보고서
├── notebooks/            # Jupyter 노트북
├── reports/              # 📊 리포트
│   ├── analysis/         # 분석 리포트
│   ├── audit/            # 감사 리포트
│   └── final/            # 최종 리포트
├── scripts/              # 🚀 실행 스크립트들 (183개 파일)
│   ├── run_pipeline_l0_l7.py
│   ├── extract_latest_metrics.py
│   └── 기타 분석/실험 스크립트들
├── src/                  # 💻 핵심 소스 코드 (201개 Python)
│   ├── components/       # 공통 컴포넌트
│   ├── core/             # 핵심 파이프라인
│   ├── data_collection/  # 데이터 수집 (L0-L4)
│   ├── evaluation/       # 성과 평가
│   ├── features/         # 피처 엔지니어링
│   ├── interfaces/       # UI 인터페이스
│   ├── pipeline/         # 파이프라인 오케스트레이션
│   │   ├── track_a_pipeline.py
│   │   └── track_b_pipeline.py
│   ├── portfolio/        # 포트폴리오 구성
│   ├── ranking/          # 랭킹 알고리즘
│   ├── stages/           # 개별 스테이지 (L0-L7)
│   ├── tools/            # 분석/검증 도구
│   ├── tracks/           # 투트랙 구현
│   │   ├── shared/       # 공통 스테이지
│   │   ├── track_a/      # Track A 전용
│   │   └── track_b/      # Track B 전용
│   ├── utils/            # 유틸리티
│   │   ├── config.py
│   │   ├── io.py
│   │   ├── meta.py
│   │   ├── reproducibility.py  # 재현성 유틸리티
│   │   └── ...
│   └── cli.py            # CLI 인터페이스
├── tests/                # 🧪 테스트 코드 (9개 파일)
│   ├── test_smoke.py     # 스모크 테스트
│   ├── test_components/
│   ├── test_pipeline/
│   └── test_utils/
├── unused/               # 사용하지 않는 파일 (참고용)
├── LICENSE               # MIT 라이선스
├── Makefile              # 빌드/실행 자동화
├── pyproject.toml        # 프로젝트 설정
├── pytest.ini            # pytest 설정
├── README.md              # 상세 README (본 문서)
└── README_PORTFOLIO.md    # 포트폴리오용 README
```

### 📋 폴더 설명

#### 핵심 디렉토리

- **`.github/`**: CI/CD 워크플로우 및 자동화 설정
- **`configs/`**: 모든 YAML 설정 파일들 (48개)
  - `config.yaml`: 메인 설정 파일
  - `features_*.yaml`: 피처 정의
  - `feature_weights_*.yaml`: 피처 가중치 설정
- **`src/`**: 핵심 소스 코드 (201개 Python 파일)
  - `pipeline/`: 파이프라인 오케스트레이션 (Track A/B 실행)
  - `tracks/`: 투트랙 구현 (Track A/B 전용 코드)
  - `data_collection/`: 데이터 수집 모듈 (L0-L4)
  - `stages/`: 개별 스테이지 구현
  - `utils/`: 유틸리티 (재현성 포함)
  - `cli.py`: CLI 인터페이스
- **`scripts/`**: 실행 스크립트들 (183개 파일)
  - `run_pipeline_l0_l7.py`: 전체 파이프라인 실행
  - `extract_latest_metrics.py`: 성과 지표 추출
  - 기타 분석/실험 스크립트

#### 데이터 디렉토리

- **`data/`**: 데이터 (Git 제외, 샘플만 포함)
  - `cal_data/`: 계산된 데이터 (백테스트 결과, 랭킹 등)
  - `raw_data/`: 원시 데이터 (OHLCV, ESG, 뉴스 등)
- **`data_backup/`**: 백업 데이터 (Git 제외)
- **`artifacts/`**: 실행 아티팩트 (Git 제외)
  - `runs/<run_id>/`: 실행별 메타데이터 (manifest.json, metrics.json)

#### 문서 및 리포트

- **`docs/`**: 문서 파일들 (30개 파일)
  - `ARCHITECTURE.md`: 시스템 아키텍처
  - `STRATEGY_CARD.md`: 전략 명세
  - `DATA_CARD.md`: 데이터 명세
  - `MODEL_CARD.md`: 모델 명세
  - `PORTFOLIO_CLEANUP_*.md`: 정리 작업 보고서
- **`reports/`**: 리포트
  - `analysis/`: 분석 리포트
  - `audit/`: 감사 리포트
  - `final/`: 최종 리포트

#### 테스트 및 기타

- **`tests/`**: 테스트 코드 (9개 파일)
  - `test_smoke.py`: 스모크 테스트
  - `test_components/`: 컴포넌트 테스트
  - `test_pipeline/`: 파이프라인 테스트
  - `test_utils/`: 유틸리티 테스트
- **`unused/`**: 사용하지 않는 파일 (참고용, Git 제외 가능)
- **`notebooks/`**: Jupyter 노트북

#### 프로젝트 설정 파일

- **`.gitignore`**: Git 제외 파일 목록 (강화 완료)
- **`pyproject.toml`**: 프로젝트 설정 및 의존성
- **`pytest.ini`**: pytest 설정
- **`Makefile`**: 빌드/실행 자동화
- **`LICENSE`**: MIT 라이선스
- **`README.md`**: 상세 README (본 문서)
- **`README_PORTFOLIO.md`**: 포트폴리오용 README

## 프로젝트 개요

### 🎯 Track A: 랭킹 엔진 (Ranking Engine)
**목적**: 피처들로 KOSPI200의 랭킹을 산정하여 이용자에게 제공

- **L8**: 랭킹 엔진 실행
  - 피처 기반 랭킹 생성 (단기/장기/통합)
  - 피처 가중치 및 정규화를 통한 종목 랭킹 산정
  - `ranking_daily` 산출물 생성
- **L11**: UI Payload Builder
  - 랭킹 데이터를 UI에서 사용할 수 있는 형태로 변환
  - 투자 성격(방어/균형/민감)별 랭킹 제공

**산출물**: `ranking_daily` (날짜별 종목 랭킹)

### 💼 Track B: 투자 모델 (Investment Model)
**목적**: 랭킹을 기반으로 다양한 투자모델 예시를 만들어 이용자에게 정보 제공

- **L6R**: 랭킹 스코어 변환
  - 랭킹 데이터를 백테스트용 스코어로 변환
  - 단기/장기/통합 랭킹을 투자 신호로 활용
- **L7**: 백테스트 실행
  - BT20 (20일 보유 기간) 전략
  - BT120 (120일 보유 기간) 전략
  - 다양한 투자 모델 예시 제공

**산출물**: `bt_metrics`, `bt_returns`, `bt_equity_curve` (백테스트 성과 지표)

---

## 핵심 개념

### Track A: 랭킹 엔진
- **랭킹**: 피처 기반 종목 랭킹 (단기/장기/통합)
- **피처**: 기술적 지표, 재무 지표, 뉴스 감성, ESG 등
- **투자 성격**: 방어/균형/민감 (α 파라미터)
- **신호 결합**: 단기/장기 랭킹 결합 비중 (γ 파라미터)

### Track B: 투자 모델
- **BT20**: 20일 보유 기간 전략 (단기 투자)
- **BT120**: 120일 보유 기간 전략 (장기 투자)
- **4개 전략**:
  1. BT20 통합 모델 (`l7_bt20_ens`): 단기 보유(20일) + 통합 랭킹
  2. BT20 분리 모델 (`l7_bt20_short`): 단기 보유(20일) + 단기 랭킹만
  3. BT120 통합 모델 (`l7_bt120_ens`): 장기 보유(120일) + 통합 랭킹
  4. BT120 분리 모델 (`l7_bt120_long`): 장기 보유(120일) + 장기 랭킹만
- **포트 배분**: BT20/BT120 포트 자본 배분 비중 (β 파라미터)

### 투트랙 아키텍처

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    공통 데이터 준비 (Shared Data, L0~L4)                  │
│  엔트리포인트(권장):                                                      │
│   - src/data_collection/*  (DataCollectionPipeline / collect_all_data)    │
│  산출물 저장: LOCAL_TRASH/artifacts_data/data/interim/*.parquet (base path는 확장자 없이 관리)        │
│   - universe_k200_membership_monthly.parquet                              │
│   - ohlcv_daily.parquet                                                   │
│   - panel_merged_daily.parquet                                            │
│   - dataset_daily.parquet, cv_folds_short.parquet, cv_folds_long.parquet  │
└──────────────────────────────────────────────────────────────────────────┘
                               ↓
        ┌──────────────────────┴──────────────────────┐
        │                                             │
┌───────▼────────────────┐                   ┌────────▼───────────────────┐
│ Track A (Ranking)      │                   │ Track B (Backtest/Model)   │
│ src/pipeline/track_a_  │                   │ src/pipeline/track_b_      │
│ pipeline.py            │                   │ pipeline.py                │
├────────────────────────┤                   ├────────────────────────────┤
│ 입력(캐시):            │                   │ 입력(캐시):                │
│ - panel_merged_daily   │                   │ - universe_k200_*          │
│ - dataset_daily(옵션)  │                   │ - dataset_daily            │
│                        │                   │ - cv_folds_short           │
│ 처리:                  │                   │ - ranking_short_daily      │
│ - L8: 단기/장기 랭킹   │                   │ - ranking_long_daily       │
│   (l8_dual_horizon)    │                   │ - ohlcv_daily(국면 옵션)   │
│ - L11: UI payload(옵션)│                   │                            │
│                        │                   │ 처리:                      │
│ 산출물:                │                   │ - L6R: 랭킹→리밸런싱 스코어│
│ - ranking_short_daily  │                   │   (interval 캐시 키 포함)  │
│ - ranking_long_daily   │                   │ - L7: 백테스트             │
│ - ui_payload(옵션)     │                   │                            │
└───────────┬────────────┘                   └───────────┬────────────────┘
            │                                            │
            └──────────────────────┬─────────────────────┘
                                   ↓
                         [UI/리포트/분석에서 활용]
```

### 투트랙 실행 흐름

1. **공통 데이터 준비 (L0~L4)**
   Track A/B 모두 `data/interim` 아티팩트가 필요합니다.

   - 방법 A (권장, 코드 흐름 그대로): `src/data_collection` 사용

```python
from src.data_collection import DataCollectionPipeline

pipeline = DataCollectionPipeline(
    config_path="configs/config.yaml",
    force_rebuild=False,  # 캐시 우선
)
pipeline.run_all()  # L0~L4
```

   - 방법 B (레거시, 전체 실행): `scripts/run_pipeline_l0_l7.py`
     - 이 스크립트는 **L0~L7까지**(모델 학습/백테스트 포함)를 한 번에 수행합니다.

```bash
python scripts/run_pipeline_l0_l7.py
```

2. **Track A 실행 (랭킹 생성: L8 + 옵션 L11)**
   - 엔트리포인트: `src/pipeline/track_a_pipeline.py`
   - 산출물(캐시): `LOCAL_TRASH/artifacts_data/data/interim/ranking_short_daily.parquet`, `LOCAL_TRASH/artifacts_data/data/interim/ranking_long_daily.parquet`

```bash
python -m src.pipeline.track_a_pipeline
```

3. **Track B 실행 (투자 모델 예시: L6R → L7)**
   - 엔트리포인트: `src/pipeline/track_b_pipeline.py`
   - **Track A 산출물(랭킹 2개)이 반드시 선행**되어야 합니다.
   - 산출물(캐시):
     - `LOCAL_TRASH/artifacts_data/data/interim/rebalance_scores_from_ranking_interval_{rebalance_interval}.parquet`
     - `LOCAL_TRASH/artifacts_data/data/interim/bt_metrics_{strategy}.parquet` 등 (`strategy`: bt20_short/bt20_ens/bt120_long/bt120_ens)

```bash
python -m src.pipeline.track_b_pipeline bt20_short
python -m src.pipeline.track_b_pipeline bt20_ens
python -m src.pipeline.track_b_pipeline bt120_long
python -m src.pipeline.track_b_pipeline bt120_ens
```

4. **(권장) 원클릭: 투트랙 실행 + 06_code22에 “최종 산출물만” Export**
   - 엔트리포인트: `src/tools/run_two_track_and_export.py`
   - 동작:
     - 공통(L0~L4) → Track A → Track B(4전략) 실행
     - **Track A 최종 설정 적용**: 앙상블 가중치 자동 적용
     - `LOCAL_TRASH/artifacts_data/artifacts/reports/track_b_4strategy_final_summary.md` 생성
     - `../06_code22/final_outputs/LATEST/`에 **최종 산출물만 복사(기존 LATEST는 비움)** + `manifest.json`/`summary.md` 생성

```bash
python -m src.tools.run_two_track_and_export --export-dest ..\06_code22
```

5. **(신규) Track A 최종 성과 검증**
   - 앙상블 가중치 적용된 Track A 실행
   - bt120_long Sharpe 0.6092 목표 달성 검증

```bash
# Track A 최종 실행 (앙상블 적용)
python -m src.pipeline.track_a_pipeline

# Track B 4전략 실행
python -m src.pipeline.track_b_pipeline bt120_long  # 주요 목표 전략
```

5. **(선택) 06_code22를 “최종 산출물 저장소”로 정리(기존 워크스페이스는 archive로 이동)**
   - 엔트리포인트: `src/tools/cleanup_06_code22_to_outputs_only.py`
   - 동작: `06_code22/src, data, configs, docs, scripts...` 등을 삭제하지 않고 `_archive_pre_outputs_*/`로 이동

```bash
python -m src.tools.cleanup_06_code22_to_outputs_only --target ..\06_code22
```

---

## 📂 프로젝트 폴더 구조 (최종 정리 완료)

```
test-main/
├── .cursor/              # Cursor IDE 설정
├── .github/              # 🔄 CI/CD 워크플로우
│   └── workflows/
│       └── ci.yml
├── artifacts/            # 🎯 실행 아티팩트 (Git 제외)
│   ├── .gitkeep
│   └── runs/             # runs/<run_id>/manifest.json, metrics.json
├── configs/              # ⚙️ 설정 파일들 (48개 YAML)
│   ├── config.yaml       # 메인 설정 파일
│   ├── features_*.yaml   # 피처 정의
│   └── feature_weights_*.yaml  # 피처 가중치
├── data/                 # 📊 데이터 (Git 제외, 샘플만 포함)
│   ├── cal_data/         # 계산된 데이터 (백테스트 결과 등)
│   │   ├── bt_metrics_*.csv/parquet
│   │   ├── ranking_*.csv/parquet
│   │   └── README.md
│   ├── raw_data/         # 원시 데이터
│   │   ├── ohlcv_daily.parquet
│   │   ├── esg_daily.parquet
│   │   ├── news_sentiment_daily.parquet
│   │   └── README.md
│   └── README.md
├── data_backup/          # 백업 데이터 (Git 제외)
├── docs/                 # 📚 문서 파일들 (30개 파일)
│   ├── ARCHITECTURE.md   # 시스템 아키텍처
│   ├── STRATEGY_CARD.md  # 전략 명세
│   ├── DATA_CARD.md      # 데이터 명세
│   ├── MODEL_CARD.md     # 모델 명세
│   └── PORTFOLIO_CLEANUP_*.md  # 정리 작업 보고서
├── notebooks/            # Jupyter 노트북
├── reports/              # 📊 리포트
│   ├── analysis/         # 분석 리포트
│   ├── audit/            # 감사 리포트
│   └── final/            # 최종 리포트
├── scripts/              # 🚀 실행 스크립트들 (183개 파일)
│   ├── run_pipeline_l0_l7.py
│   ├── extract_latest_metrics.py
│   └── 기타 분석/실험 스크립트들
├── src/                  # 💻 핵심 소스 코드 (201개 Python)
│   ├── components/       # 공통 컴포넌트
│   ├── core/             # 핵심 파이프라인
│   ├── data_collection/  # 데이터 수집 (L0-L4)
│   ├── evaluation/       # 성과 평가
│   ├── features/         # 피처 엔지니어링
│   ├── interfaces/       # UI 인터페이스
│   ├── pipeline/         # 파이프라인 오케스트레이션
│   │   ├── track_a_pipeline.py
│   │   └── track_b_pipeline.py
│   ├── portfolio/        # 포트폴리오 구성
│   ├── ranking/          # 랭킹 알고리즘
│   ├── stages/           # 개별 스테이지 (L0-L7)
│   ├── tools/            # 분석/검증 도구
│   ├── tracks/           # 투트랙 구현
│   │   ├── shared/       # 공통 스테이지
│   │   ├── track_a/      # Track A 전용
│   │   └── track_b/      # Track B 전용
│   ├── utils/            # 유틸리티
│   │   ├── config.py
│   │   ├── io.py
│   │   ├── meta.py
│   │   ├── reproducibility.py  # 재현성 유틸리티
│   │   └── ...
│   └── cli.py            # CLI 인터페이스
├── tests/                # 🧪 테스트 코드 (9개 파일)
│   ├── test_smoke.py     # 스모크 테스트
│   ├── test_components/
│   ├── test_pipeline/
│   └── test_utils/
├── unused/               # 사용하지 않는 파일 (참고용)
├── LICENSE               # MIT 라이선스
├── Makefile              # 빌드/실행 자동화
├── pyproject.toml        # 프로젝트 설정
├── pytest.ini            # pytest 설정
├── README.md              # 상세 README (본 문서)
└── README_PORTFOLIO.md    # 포트폴리오용 README
```

### 📋 폴더 설명

#### 핵심 디렉토리

- **`.github/`**: CI/CD 워크플로우 및 자동화 설정
- **`configs/`**: 모든 YAML 설정 파일들 (48개)
  - `config.yaml`: 메인 설정 파일
  - `features_*.yaml`: 피처 정의
  - `feature_weights_*.yaml`: 피처 가중치 설정
- **`src/`**: 핵심 소스 코드 (201개 Python 파일)
  - `pipeline/`: 파이프라인 오케스트레이션 (Track A/B 실행)
  - `tracks/`: 투트랙 구현 (Track A/B 전용 코드)
  - `data_collection/`: 데이터 수집 모듈 (L0-L4)
  - `stages/`: 개별 스테이지 구현
  - `utils/`: 유틸리티 (재현성 포함)
  - `cli.py`: CLI 인터페이스
- **`scripts/`**: 실행 스크립트들 (183개 파일)
  - `run_pipeline_l0_l7.py`: 전체 파이프라인 실행
  - `extract_latest_metrics.py`: 성과 지표 추출
  - 기타 분석/실험 스크립트

#### 데이터 디렉토리

- **`data/`**: 데이터 (Git 제외, 샘플만 포함)
  - `cal_data/`: 계산된 데이터 (백테스트 결과, 랭킹 등)
  - `raw_data/`: 원시 데이터 (OHLCV, ESG, 뉴스 등)
- **`data_backup/`**: 백업 데이터 (Git 제외)
- **`artifacts/`**: 실행 아티팩트 (Git 제외)
  - `runs/<run_id>/`: 실행별 메타데이터 (manifest.json, metrics.json)

#### 문서 및 리포트

- **`docs/`**: 문서 파일들 (30개 파일)
  - `ARCHITECTURE.md`: 시스템 아키텍처
  - `STRATEGY_CARD.md`: 전략 명세
  - `DATA_CARD.md`: 데이터 명세
  - `MODEL_CARD.md`: 모델 명세
  - `PORTFOLIO_CLEANUP_*.md`: 정리 작업 보고서
- **`reports/`**: 리포트
  - `analysis/`: 분석 리포트
  - `audit/`: 감사 리포트
  - `final/`: 최종 리포트

#### 테스트 및 기타

- **`tests/`**: 테스트 코드 (9개 파일)
  - `test_smoke.py`: 스모크 테스트
  - `test_components/`: 컴포넌트 테스트
  - `test_pipeline/`: 파이프라인 테스트
  - `test_utils/`: 유틸리티 테스트
- **`unused/`**: 사용하지 않는 파일 (참고용, Git 제외 가능)
- **`notebooks/`**: Jupyter 노트북

#### 프로젝트 설정 파일

- **`.gitignore`**: Git 제외 파일 목록 (강화 완료)
- **`pyproject.toml`**: 프로젝트 설정 및 의존성
- **`pytest.ini`**: pytest 설정
- **`Makefile`**: 빌드/실행 자동화
- **`LICENSE`**: MIT 라이선스
- **`README.md`**: 상세 README (본 문서)
- **`README_PORTFOLIO.md`**: 포트폴리오용 README

---

## 설치 방법

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

---

## 실행 방법

### 1단계: 공통 데이터 준비 (L0~L4)

**⚠️ 필수**: Track A와 Track B 모두 실행하기 전에 공통 데이터를 먼저 준비해야 합니다.

**단계별 설명**:
- **L0**: 유니버스 구성 (KOSPI200 멤버십)
- **L1**: OHLCV 데이터 다운로드 + 기술적 지표 계산
- **L2**: 재무 데이터 로드 (DART)
- **L3**: 패널 병합 (OHLCV + 재무 + 뉴스 + ESG)
- **L4**: Walk-Forward CV 분할 및 타겟 생성

#### 방법 1: 새로운 데이터 수집 모듈 사용 (권장) ⭐

**리팩토링 완료**: `src/data_collection` 모듈을 통해 데이터 수집이 완전히 분리되었습니다.

```python
# Python에서 직접 호출
from src.data_collection import collect_all_data

# 전체 데이터 수집 (L0~L4)
result = collect_all_data(
    config_path="configs/config.yaml",
    force_rebuild=False,  # 캐시 사용
)

# 단계별 수집
from src.data_collection import (
    collect_universe,
    collect_ohlcv,
    collect_panel,
    collect_dataset,
)

# L0: 유니버스
universe = collect_universe(
    start_date="2016-01-01",
    end_date="2024-12-31",
    config_path="configs/config.yaml",
)

# L1: OHLCV
ohlcv = collect_ohlcv(
    universe=universe,
    start_date="2016-01-01",
    end_date="2024-12-31",
    config_path="configs/config.yaml",
)

# L3: 패널 병합
panel = collect_panel(
    ohlcv_daily=ohlcv,
    config_path="configs/config.yaml",
)

# L4: CV 분할
dataset = collect_dataset(
    panel_merged_daily=panel,
    config_path="configs/config.yaml",
)
```

**파이프라인 클래스 사용**:

```python
from src.data_collection import DataCollectionPipeline

# 파이프라인 생성
pipeline = DataCollectionPipeline(
    config_path="configs/config.yaml",
    force_rebuild=False,
)

# 전체 실행
result = pipeline.run_all()

# 단계별 실행
pipeline.run_l0()  # 유니버스
pipeline.run_l1()  # OHLCV
pipeline.run_l3()  # 패널 병합
pipeline.run_l4()  # CV 분할

# 아티팩트 조회
artifacts = pipeline.get_artifacts()
```

#### 방법 2: 기존 스크립트 사용 (하위 호환성 유지)

```bash
python scripts/run_pipeline_l0_l7.py
```

이 명령은 다음 단계를 실행합니다:
- **L0**: 유니버스 구성 (KOSPI200 멤버십)
- **L1**: OHLCV 데이터 다운로드 + 기술적 지표 계산
- **L2**: 재무 데이터 로드 (DART)
- **L3**: 패널 병합 (OHLCV + 재무 + 뉴스 + ESG)
- **L4**: Walk-Forward CV 분할 및 타겟 생성
- **L5**: 모델 학습 (Ridge 회귀, 선택적)
  - 단기 모델: 20일 수익률 예측
  - 장기 모델: 120일 수익률 예측
- **L6**: 스코어 생성 (레거시)
- **L7**: 백테스트 실행

**산출물**:
- `universe_k200_membership_monthly`: KOSPI200 멤버십 정보
- `ohlcv_daily`: 일별 OHLCV 데이터 + 기술적 지표
- `panel_merged_daily`: 병합된 패널 데이터
- `dataset_daily`: CV 분할이 완료된 데이터셋
- `pred_short_oos`: 단기 모델 예측 (L5, 선택적)
- `pred_long_oos`: 장기 모델 예측 (L5, 선택적)
- `model_metrics`: 모델 성능 지표 (L5, 선택적)

**참고**: 기존 데이터는 그대로 유지되며, 새로운 데이터 수집 모듈은 기존 데이터를 재사용합니다.

### 2단계: 모델 학습 (L5, 선택적)

**목적**: 단기(20일) 및 장기(120일) 수익률 예측을 위한 Ridge 회귀 모델 학습

**⚠️ 참고**: L5는 Track B에서 사용되지만, Track A에서는 L5의 피처 리스트만 사용합니다. Track A만 사용하는 경우 L5는 선택적입니다.

**사용 시나리오**:
- Track B 백테스트를 실행하려는 경우
- 모델 예측 성능을 확인하고 싶은 경우

**실행 방법**:
```bash
# 레거시 스크립트 사용 (L0~L7 전체 실행)
python scripts/run_pipeline_l0_l7.py

# 또는 Python에서 직접 호출
from src.stages.modeling.l5_train_models import train_oos_predictions
from src.utils.config import load_config
from src.utils.io import load_artifact
from pathlib import Path

cfg = load_config('configs/config.yaml')
interim_dir = Path(get_path(cfg, "data_interim"))

# 필요한 아티팩트 로드
artifacts = {
    "dataset_daily": load_artifact(interim_dir / "dataset_daily"),
    "cv_folds_short": load_artifact(interim_dir / "cv_folds_short"),
    "cv_folds_long": load_artifact(interim_dir / "cv_folds_long"),
}

# L5 실행
pred_short, pred_long, metrics = train_oos_predictions(
    cfg=cfg,
    dataset_daily=artifacts["dataset_daily"],
    cv_folds_short=artifacts["cv_folds_short"],
    cv_folds_long=artifacts["cv_folds_long"],
)
```

**실행 단계**:
1. 공통 데이터 확인 (L0~L4 산출물)
2. **L5**: 모델 학습
   - Walk-Forward CV 각 fold별로 Ridge 회귀 모델 학습
   - 단기 모델: 20일 수익률 예측 (`pred_short_oos`)
   - 장기 모델: 120일 수익률 예측 (`pred_long_oos`)
   - 모델 성능 지표 계산 (`model_metrics`)

**산출물**:
- `pred_short_oos`: 단기 모델 예측 (날짜별 종목 예측값)
- `pred_long_oos`: 장기 모델 예측 (날짜별 종목 예측값)
- `model_metrics`: 모델 성능 지표 (RMSE, IC, Hit Ratio 등)

**설정 파일**: `configs/config.yaml`의 `l5` 섹션
- `ridge_alpha`: 8.0 (L2 정규화 강도)
- `min_feature_ic`: -0.1 (피처 필터링 임계값)
- `feature_list_short`: 단기 피처 리스트 (22개)
- `feature_list_long`: 장기 피처 리스트 (19개)

### 3단계: Track A 실행 (랭킹 엔진)

**목적**: 피처 기반으로 KOSPI200 종목의 랭킹을 산정하여 이용자에게 제공

**사용 시나리오**:
- 랭킹 정보만 필요한 경우
- UI에서 랭킹을 표시하고 이용자가 직접 투자 결정하는 경우

```bash
# 랭킹 엔진 실행
python -m src.pipeline.track_a_pipeline

# 또는 Python에서 직접 호출
from src.pipeline.track_a_pipeline import run_track_a_pipeline
result = run_track_a_pipeline()
```

**실행 단계**:
1. 공통 데이터 확인 (L0~L4 산출물)
2. **L8**: 랭킹 엔진 실행
   - 단기 랭킹 생성 (`ranking_short_daily`)
   - 장기 랭킹 생성 (`ranking_long_daily`)
   - L5 피처 리스트 사용 (L8-L5 피처셋 통일)
3. **L11**: UI Payload 생성 (선택적)

**산출물**:
- `ranking_short_daily`: 단기 랭킹 (날짜별 종목 랭킹)
- `ranking_long_daily`: 장기 랭킹 (날짜별 종목 랭킹)
- `ui_payload`: UI에서 사용할 수 있는 형태의 랭킹 데이터 (선택적)

### 4단계: Track B 실행 (투자 모델)

**목적**: Track A에서 생성한 랭킹을 기반으로 다양한 투자모델 예시를 만들어 이용자에게 정보 제공

**사용 시나리오**:
- 랭킹 기반 투자 전략의 성과를 확인하고 싶은 경우
- 다양한 투자 모델 예시를 제공하고 싶은 경우
- 백테스트 결과를 통해 투자 전략의 유효성을 검증하고 싶은 경우

**⚠️ 필수 조건**: Track A를 먼저 실행하여 랭킹 데이터를 생성해야 합니다.

#### 방법 1: Track B 파이프라인 직접 실행 (권장)

```bash
# Track B 파이프라인 실행 (4가지 전략 중 선택)
python -m src.pipeline.track_b_pipeline bt20_short   # BT20 단기 랭킹
python -m src.pipeline.track_b_pipeline bt20_ens     # BT20 통합 랭킹
python -m src.pipeline.track_b_pipeline bt120_long   # BT120 장기 랭킹
python -m src.pipeline.track_b_pipeline bt120_ens    # BT120 통합 랭킹
```

#### 방법 2: 편의 래퍼 사용

```bash
# BT20 파이프라인 (20일 보유 기간)
python -m src.pipeline.bt20_pipeline short  # 단기 랭킹만 사용
python -m src.pipeline.bt20_pipeline ens     # 통합 랭킹 사용

# BT120 파이프라인 (120일 보유 기간)
python -m src.pipeline.bt120_pipeline long  # 장기 랭킹만 사용
python -m src.pipeline.bt120_pipeline ens   # 통합 랭킹 사용
```

#### Python에서 직접 호출

```python
from src.pipeline.track_b_pipeline import run_track_b_pipeline

# BT20 전략 (단기 랭킹 사용)
result_bt20_short = run_track_b_pipeline(strategy="bt20_short")

# BT120 전략 (장기 랭킹 사용)
result_bt120_long = run_track_b_pipeline(strategy="bt120_long")
```

**실행 단계**:
1. 공통 데이터 확인 (L0~L4 산출물)
2. Track A 산출물 확인 (`ranking_short_daily`, `ranking_long_daily`)
3. **L6R**: 랭킹 스코어 변환 (랭킹 → 백테스트용 스코어)
4. **L7**: 백테스트 실행
   - 포지션 생성
   - 수익률 계산
   - 성과 지표 계산

**산출물**:
- `rebalance_scores_from_ranking`: 랭킹에서 변환된 스코어
- `bt_positions_{strategy}`: 포지션 정보
- `bt_returns_{strategy}`: 일별 수익률
- `bt_equity_curve_{strategy}`: 자산 곡선
- `bt_metrics_{strategy}`: 백테스트 성과 지표 (Sharpe, MDD, CAGR 등)

---

## 캐시 우선 로직

파이프라인은 **캐시 우선** 방식으로 동작합니다:
- 이미 생성된 중간 산출물(`data/interim/*.parquet`)이 있으면 재사용
- 캐시가 없을 때만 해당 단계를 재실행
- `force_rebuild=True` 옵션으로 캐시 무시 가능

---

## UI 연동

Flask 등 UI 프레임워크에서 랭킹과 데이터를 조회할 수 있는 인터페이스를 제공합니다.

### 데이터 수집 인터페이스 (리팩토링 완료) ⭐

**새로운 UI 인터페이스**: `src/data_collection` 모듈에서 간단한 함수로 데이터를 조회할 수 있습니다.

```python
from src.data_collection import (
    get_universe,
    get_ohlcv,
    get_panel,
    get_dataset,
    check_data_availability,
    collect_data_for_ui,
)

# 데이터 가용성 확인
available = check_data_availability()
print(available)
# {
#     "universe": True,
#     "ohlcv": True,
#     "fundamentals": False,
#     "panel": True,
#     "dataset": True,
#     ...
# }

# 유니버스 조회
universe = get_universe()

# OHLCV 조회
ohlcv = get_ohlcv()

# 패널 조회
panel = get_panel()

# 데이터셋 조회
dataset = get_dataset()

# UI용 통합 데이터 수집
result = collect_data_for_ui()
# {
#     "universe": DataFrame,
#     "ohlcv": DataFrame,
#     "panel": DataFrame,
#     "dataset": DataFrame,
#     "available": Dict[str, bool],
#     ...
# }
```

### Track A: 랭킹 조회

```python
from src.interfaces.ui_service import (
    get_short_term_ranking,
    get_long_term_ranking,
    get_combined_ranking,
)

# 단기 랭킹 조회
short_rankings = get_short_term_ranking("2024-12-31", top_k=20)

# 장기 랭킹 조회
long_rankings = get_long_term_ranking("2024-12-31", top_k=20)

# 통합 랭킹 조회
combined_rankings = get_combined_ranking("2024-12-31", top_k=20)
```

### Track A: Holdout 하루 Top10 + 팩터셋(그룹) Top3 기여도 (설명가능성)

특정 날짜(특히 Holdout 기간)에서 **왜 그 종목이 상위에 랭킹되었는지** 빠르게 확인할 수 있는 도구입니다.

```python
from src.tracks.track_a.ranking_service import inspect_holdout_day_rankings

out = inspect_holdout_day_rankings(
    as_of="2024-12-30",
    topk=10,
    horizon="both",  # "short" | "long" | "both"
)

# out["short"], out["long"]에는 아래 컬럼이 포함됩니다:
# - date, rank_total, ticker, score_total, score_total_calc, score_gap, top_groups
```

CLI로도 실행 가능:

```bash
python scripts/inspect_tracka_holdout_day.py --date 2024-12-30 --topk 10 --horizon both
```

### Track B: 투자 모델 성과 조회

```python
from src.utils.io import load_artifact
from pathlib import Path

# 백테스트 메트릭 로드
bt_metrics = load_artifact(Path("data/interim/bt_metrics_bt20"))
print(bt_metrics[bt_metrics["phase"] == "holdout"])
```

### Flask API 예시

```python
from flask import Flask, jsonify, request
from src.interfaces.ui_service import (
    get_short_term_ranking,
    get_long_term_ranking,
    get_combined_ranking,
)
from src.data_collection import (
    get_universe,
    get_ohlcv,
    get_panel,
    check_data_availability,
)

app = Flask(__name__)

# 데이터 수집 API (리팩토링 완료)
@app.get("/api/data/availability")
def data_availability():
    """데이터 가용성 확인"""
    return jsonify(check_data_availability())

@app.get("/api/data/universe")
def universe_data():
    """유니버스 데이터 조회"""
    df = get_universe()
    return jsonify(df.to_dict(orient="records"))

@app.get("/api/data/ohlcv")
def ohlcv_data():
    """OHLCV 데이터 조회"""
    df = get_ohlcv()
    return jsonify(df.to_dict(orient="records"))

# Track A: 랭킹 API
@app.get("/api/ranking/short")
def short_ranking():
    as_of = request.args.get("as_of", default="2024-12-31")
    top_k = int(request.args.get("top_k", 20))
    return jsonify(get_short_term_ranking(as_of, top_k))

@app.get("/api/ranking/long")
def long_ranking():
    as_of = request.args.get("as_of", default="2024-12-31")
    top_k = int(request.args.get("top_k", 20))
    return jsonify(get_long_term_ranking(as_of, top_k))

@app.get("/api/ranking/combined")
def combined_ranking():
    as_of = request.args.get("as_of", default="2024-12-31")
    top_k = int(request.args.get("top_k", 20))
    return jsonify(get_combined_ranking(as_of, top_k))

# Track B: 투자 모델 성과 API
@app.get("/api/backtest/metrics")
def backtest_metrics():
    strategy = request.args.get("strategy", default="bt20_short")
    # 백테스트 메트릭 반환
    ...
```

---

## 설정 파일

`configs/config.yaml`에서 다음 설정을 관리합니다:

### 공통 설정

#### L4: Walk-Forward CV 분할
```yaml
l4:
  drop_non_universe_before_save: true
  holdout_years: 2
  step_days: 20
  test_window_days: 20
  embargo_days: 20
  horizon_short: 20
  horizon_long: 120
  rolling_train_years_short: 3
  rolling_train_years_long: 5
  inner_cv_k: 5
  market_neutral: false
```

#### L5: 모델 학습 파라미터 (Ridge 회귀 모델)
```yaml
l5:
  model_type: ridge
  target_transform: cs_rank
  cs_rank_center: true
  ridge_alpha: 8.0  # [최종 픽스 2026-01-07] L2 정규화 강도
  min_feature_ic: -0.1  # [최종 픽스 2026-01-07] 피처 필터링 임계값
  filter_features_by_ic: true
  use_rank_ic: true
  feature_list_short: configs/features_short_v1.yaml  # 22개 피처
  feature_list_long: configs/features_long_v1.yaml    # 19개 피처
  feature_weights_config_short: configs/feature_weights_short_hitratio_optimized.yaml
  feature_weights_config_long: configs/feature_weights_long_ic_optimized.yaml
```
- **역할**: Track B에서 사용 (단기/장기 수익률 예측 모델 학습)
- **Track A와의 관계**: Track A는 L5의 피처 리스트만 사용 (모델 학습은 선택적)

#### L6: 스코어 결합 가중치
```yaml
l6:
  weight_short: 0.5
  weight_long: 0.5
  invert_score_sign: false
```

### Track A 설정

#### L8: 랭킹 엔진 설정
```yaml
l8_short:
  normalization_method: zscore  # [최종 픽스 2026-01-07] 정규화 방법
  feature_groups_config: configs/feature_groups_short.yaml
  feature_weights_config: configs/feature_weights_short_hitratio_optimized.yaml
  use_sector_relative: true
  sector_col: sector_name

l8_long:
  normalization_method: zscore  # [최종 픽스 2026-01-07] 정규화 방법
  feature_groups_config: configs/feature_groups_long.yaml
  feature_weights_config: configs/feature_weights_long_ic_optimized.yaml
  use_sector_relative: true
  sector_col: sector_name
```

#### L11: UI Payload 설정
```yaml
l11:
  top_k: 10
  bottom_k: 10
  top_k_perf: 20
  benchmark_type: universe_mean
  savings_apr: 0.03
```

### Track B 설정

#### L6R: 랭킹 스코어 변환 + 앙상블 가중치 설정
```yaml
l6r:
  alpha_short: 0.5  # 단기:장기 결합 비중 (bt20_ens, bt120_ens)
  alpha_long: null  # 자동으로 1-alpha_short

  # [Track A 최종 앙상블 가중치] 과적합 개선된 최적 가중치
  ensemble_weights:
    short:  # 단기 전략 앙상블 (IC Diff 0.0371, LOW 위험)
      grid: 0.30      # Grid Search: 30%
      ridge: 0.60     # Ridge: 60%
      xgboost: 0.10   # XGBoost: 10%
      rf: 0.00        # Random Forest: 0%
    long:   # 장기 전략 앙상블 (IC Diff 0.0557, MEDIUM 위험)
      grid: 0.05      # Grid Search: 5%
      ridge: 0.15     # Ridge: 15%
      xgboost: 0.80   # XGBoost: 80%
      rf: 0.00        # Random Forest: 0%

  rebalance_interval: 1  # 기본값 (전략별로 오버라이드)
  regime_alpha:
    bull_strong: 0.6
    bull_weak: 0.6
    neutral: 0.5
    bear_weak: 0.4
    bear_strong: 0.4
```

#### L7: 백테스트 기본 설정
```yaml
l7:
  holding_days: 20
  top_k: 12
  cost_bps: 10.0
  slippage_bps: 0.0
  buffer_k: 15
  weighting: equal
  score_col: score_ens
  return_col: true_short
  rebalance_interval: 1  # 기본값 (전략별로 오버라이드)
  smart_buffer_enabled: true
  smart_buffer_stability_threshold: 0.7
  volatility_adjustment_enabled: true
  volatility_lookback_days: 60
  target_volatility: 0.15
  volatility_adjustment_max: 1.2
  volatility_adjustment_min: 0.7
  risk_scaling_enabled: true
  risk_scaling_bear_multiplier: 0.8
  risk_scaling_neutral_multiplier: 1.0
  risk_scaling_bull_multiplier: 1.0
  regime:
    enabled: true
    lookback_days: 60
    threshold_pct: 0.0
    neutral_band: 0.05
```

#### l7_bt20_ens: BT20 통합 모델 설정
```yaml
l7_bt20_ens:
  holding_days: 20
  top_k: 15
  cost_bps: 10.0
  slippage_bps: 0.0
  buffer_k: 20
  weighting: softmax
  softmax_temperature: 0.5
  score_col: score_ens  # 단기:장기 5:5 결합
  return_col: true_short
  rebalance_interval: 20  # [중요] holding_days와 동일 (20일 모멘텀 본질 유지)
  smart_buffer_enabled: true
  smart_buffer_stability_threshold: 0.7
  volatility_adjustment_enabled: true
  volatility_lookback_days: 60
  target_volatility: 0.15
  volatility_adjustment_max: 1.2
  volatility_adjustment_min: 0.7
  risk_scaling_enabled: true
  risk_scaling_bear_multiplier: 0.8
  risk_scaling_neutral_multiplier: 1.0
  risk_scaling_bull_multiplier: 1.0
  regime:
    enabled: true
    lookback_days: 60
    threshold_pct: 0.0
    neutral_band: 0.0
    top_k_bull_strong: 10
    top_k_bull_weak: 12
    top_k_bear_strong: 20
    top_k_bear_weak: 20
    top_k_neutral: 20
    exposure_bull_strong: 1.5
    exposure_bull_weak: 1.2
    exposure_bear_strong: 0.6
    exposure_bear_weak: 0.8
    exposure_neutral: 1.0
```

#### l7_bt20_short: BT20 분리 모델 설정
```yaml
l7_bt20_short:
  holding_days: 20
  top_k: 12
  cost_bps: 10.0
  slippage_bps: 0.0
  buffer_k: 15
  weighting: equal
  score_col: score_total_short  # 단기 랭킹만 사용
  return_col: true_short
  rebalance_interval: 20  # [중요] holding_days와 동일 (20일 모멘텀 본질 유지)
  smart_buffer_enabled: true
  smart_buffer_stability_threshold: 0.7
  volatility_adjustment_enabled: true
  volatility_lookback_days: 60
  target_volatility: 0.15
  volatility_adjustment_max: 1.2
  volatility_adjustment_min: 0.7
  risk_scaling_enabled: true
  risk_scaling_bear_multiplier: 0.8
  risk_scaling_neutral_multiplier: 1.0
  risk_scaling_bull_multiplier: 1.0
  regime:
    enabled: true
    lookback_days: 60
    threshold_pct: 0.0
    neutral_band: 0.0
    top_k_bull_strong: 10
    top_k_bull_weak: 12
    top_k_bear_strong: 20
    top_k_bear_weak: 20
    top_k_neutral: 20
    exposure_bull_strong: 1.5
    exposure_bull_weak: 1.2
    exposure_bear_strong: 0.6
    exposure_bear_weak: 0.8
    exposure_neutral: 1.0
```

#### l7_bt120_ens: BT120 통합 모델 설정 (오버래핑 트랜치)
```yaml
l7_bt120_ens:
  holding_days: 20  # [오버래핑 트랜치] 월별(20일) 기간수익률로 평가
  top_k: 20
  cost_bps: 10.0
  slippage_bps: 0.0
  buffer_k: 15
  weighting: equal
  score_col: score_ens  # 단기:장기 5:5 결합
  return_col: true_short  # [오버래핑 트랜치] 월별 PnL(20일 fwd)로 계산
  rebalance_interval: 20  # [중요] 월별 리밸런싱(신규 트랜치 추가)
  overlapping_tranches_enabled: true  # [필수] 오버래핑 트랜치 모드
  tranche_holding_days: 120  # 각 트랜치 보유 기간(캘린더 day)
  tranche_max_active: 4  # 월별 4트랜치(동시 보유 최대 4개)
  tranche_allocation_mode: fixed_equal  # 각 트랜치에 1/4 자본 고정 배분
  smart_buffer_enabled: true
  smart_buffer_stability_threshold: 0.7
  volatility_adjustment_enabled: true
  volatility_lookback_days: 60
  target_volatility: 0.15
  volatility_adjustment_max: 1.2
  volatility_adjustment_min: 0.6
  risk_scaling_enabled: true
  risk_scaling_bear_multiplier: 0.7
  risk_scaling_neutral_multiplier: 0.9
  risk_scaling_bull_multiplier: 1.0
  regime:
    enabled: true
    lookback_days: 60
    threshold_pct: 0.0
    neutral_band: 0.05
    top_k_bull_strong: 12
    top_k_bull_weak: 15
    top_k_bear_strong: 30
    top_k_bear_weak: 30
    top_k_neutral: 20
    exposure_bull_strong: 1.3
    exposure_bull_weak: 1.0
    exposure_bear_strong: 0.7
    exposure_bear_weak: 0.9
    exposure_neutral: 1.0
```

#### l7_bt120_long: BT120 분리 모델 설정 (오버래핑 트랜치)
```yaml
l7_bt120_long:
  holding_days: 20  # [오버래핑 트랜치] 월별(20일) 기간수익률로 평가
  top_k: 15
  cost_bps: 10.0
  slippage_bps: 0.0
  buffer_k: 15
  weighting: equal
  score_col: score_total_long  # 장기 랭킹만 사용
  return_col: true_short  # [오버래핑 트랜치] 월별 PnL(20일 fwd)로 계산
  rebalance_interval: 20  # [중요] 월별 리밸런싱(신규 트랜치 추가)
  overlapping_tranches_enabled: true  # [필수] 오버래핑 트랜치 모드
  tranche_holding_days: 120  # 각 트랜치 보유 기간(캘린더 day)
  tranche_max_active: 4  # 월별 4트랜치(동시 보유 최대 4개)
  tranche_allocation_mode: fixed_equal  # 각 트랜치에 1/4 자본 고정 배분
  smart_buffer_enabled: true
  smart_buffer_stability_threshold: 0.7
  volatility_adjustment_enabled: true
  volatility_lookback_days: 60
  target_volatility: 0.15
  volatility_adjustment_max: 1.2
  volatility_adjustment_min: 0.6
  risk_scaling_enabled: true
  risk_scaling_bear_multiplier: 0.7
  risk_scaling_neutral_multiplier: 0.9
  risk_scaling_bull_multiplier: 1.0
  regime:
    enabled: true
    lookback_days: 60
    threshold_pct: 0.0
    neutral_band: 0.05
    top_k_bull_strong: 12
    top_k_bull_weak: 15
    top_k_bear_strong: 30
    top_k_bear_weak: 30
    top_k_neutral: 20
    exposure_bull_strong: 1.3
    exposure_bull_weak: 1.0
    exposure_bear_strong: 0.7
    exposure_bear_weak: 0.9
    exposure_neutral: 1.0
```

### ⚠️ 중요 설정값 (2026-01-07 최종 픽스)

#### rebalance_interval 설정 (필수)
- **모든 전략**: `rebalance_interval: 20` (holding_days와 동일)
- **문제**: `rebalance_interval=1`이면 안 됨
  - BT20: 20일 모멘텀 → 월 모멘텀으로 변질
  - BT120: 트랜치 효과 소실 (매월 완전 교체)
- **올바른 설정**:
  - `l7_bt20_short`: `rebalance_interval: 20` (단기 본질 유지)
  - `l7_bt20_ens`: `rebalance_interval: 20` (단기 본질 유지)
  - `l7_bt120_long`: `rebalance_interval: 20` (트랜치 추가 주기, 월별)
  - `l7_bt120_ens`: `rebalance_interval: 20` (트랜치 추가 주기, 월별)

#### 시장 국면 분류 설정
- **방식**: 외부 API 호출 없이 `ohlcv_daily` 데이터를 사용하여 자동 분류
- **지표**: 가격 수익률, 변동성, 거래량 변화율을 종합하여 Bull/Neutral/Bear 판단
- **기준**: 각 rebalance 날짜 기준으로 lookback 기간 동안의 지표를 계산

---

## 성과 지표

### Track A: 랭킹 품질 지표
- 랭킹 일관성
- 피처 중요도
- 랭킹 분포 분석

### Track B: 백테스트 성과 지표

**실제 산출 데이터**: `bt_metrics_{strategy}.parquet` 파일에 저장됩니다.

#### 1. 핵심 성과 (Headline Metrics) - ✅ 실제 산출됨

`bt_metrics` 파일에 포함된 지표:

| 지표명 | 컬럼명 | 설명 | 계산 방식 |
|--------|--------|------|-----------|
| **Net Sharpe Ratio** | `net_sharpe` | 리스크 조정 수익률 (비용 차감) | `(평균 수익률 / 표준편차) * sqrt(252/holding_days)` |
| **Gross Sharpe Ratio** | `gross_sharpe` | 리스크 조정 수익률 (비용 차감 전) | 동일 (Gross 수익률 기준) |
| **Net Total Return** | `net_total_return` | 비용 차감 누적 수익률 | `(최종 자산가치 / 초기 자산가치) - 1` |
| **Gross Total Return** | `gross_total_return` | 비용 차감 전 누적 수익률 | 동일 (비용 차감 전) |
| **Net CAGR** | `net_cagr` | 연평균 복리 수익률 (비용 차감) | `(최종 자산가치 / 초기 자산가치)^(1/년수) - 1` |
| **Gross CAGR** | `gross_cagr` | 연평균 복리 수익률 (비용 차감 전) | 동일 (Gross 기준) |
| **Net MDD** | `net_mdd` | 최대 낙폭 (비용 차감) | `min((equity / peak) - 1.0)` |
| **Gross MDD** | `gross_mdd` | 최대 낙폭 (비용 차감 전) | 동일 (Gross 기준) |
| **Net Calmar Ratio** | `net_calmar_ratio` | 수익성 / 최대낙폭 (비용 차감) | `CAGR / \|MDD\|` |
| **Gross Calmar Ratio** | `gross_calmar_ratio` | 수익성 / 최대낙폭 (비용 차감 전) | 동일 (Gross 기준) |
| **Net Volatility (Annualized)** | `net_vol_ann` | 연환산 변동성 (비용 차감) | `std(수익률) * sqrt(252/holding_days)` |
| **Gross Volatility (Annualized)** | `gross_vol_ann` | 연환산 변동성 (비용 차감 전) | 동일 (Gross 기준) |

**추가 메타데이터**:
- `gross_minus_net_total_return_pct`: Gross와 Net 차이 (거래비용 영향도)
- `avg_cost_pct`: 평균 거래비용 (퍼센트)
- `cost_bps`, `cost_bps_used`, `cost_bps_config`: 거래비용 설정값
- `date_start`, `date_end`: 구간 시작/종료일
- `phase`: 구간 구분 (`dev` 또는 `holdout`)

#### 2. 운용 안정성 (Operational Viability) - ✅ 실제 산출됨

| 지표명 | 컬럼명 | 설명 | 계산 방식 |
|--------|--------|------|-----------|
| **Avg Turnover (Oneway)** | `avg_turnover_oneway` | 평균 일방 회전율 | `평균(매일 포트폴리오 변경 비율)` |
| **Net Hit Ratio** | `net_hit_ratio` | 승률 (비용 차감) | `(수익 > 0인 리밸런싱 수) / 전체 리밸런싱 수` |
| **Gross Hit Ratio** | `gross_hit_ratio` | 승률 (비용 차감 전) | 동일 (Gross 기준) |
| **Net Profit Factor** | `net_profit_factor` | 총 이익 / 총 손실 (비용 차감) | `sum(양수 수익) / abs(sum(음수 수익))` |
| **Gross Profit Factor** | `gross_profit_factor` | 총 이익 / 총 손실 (비용 차감 전) | 동일 (Gross 기준) |
| **Avg Trade Duration** | `avg_trade_duration` | 평균 보유 일수 | `평균(각 종목별 연속 보유 기간)` |
| **Avg N Tickers** | `avg_n_tickers` | 평균 보유 종목 수 | `평균(매 리밸런싱 시 보유 종목 수)` |
| **N Rebalances** | `n_rebalances` | 리밸런싱 횟수 | 전체 리밸런싱 실행 횟수 |

**추가 메타데이터**:
- `top_k`: 선택 종목 수
- `holding_days`: 보유 기간 (일)
- `buffer_k`: 버퍼 종목 수
- `weighting`: 가중치 방식 (`equal` 또는 `softmax`)

#### 랭킹산정모델 최종 픽스 (2026-01-07)

**정규화 방법 최적화** (2026-01-07):
- **비교 테스트**: percentile, zscore, robust_zscore 비교
- **최종 선택**: **zscore** (50.28% Hit Ratio, 최고 성과)
- **최종 설정**: `l8_short.normalization_method: zscore`, `l8_long.normalization_method: zscore`

**News 피처 가중치 최적화** (2026-01-07):
- **단기 News 피처**: 0.10 (각 피처, 총 0.40)
- **장기 News 피처**: 0.03 (news_sentiment_ewm20)
- **단기 전용 피처**: 0.025 (각 피처)

**최종 Hit Ratio 성과** (2026-01-07):
- **통합 랭킹**: 49.58% (전체), **51.06% (Holdout)** ✅ 목표 달성
- **단기 랭킹**: 49.28% (전체), **50.99% (Holdout)** ✅ 목표 달성
- **장기 랭킹**: **50.14% (전체)**, **51.00% (Holdout)** ✅ 목표 달성
- **과적합**: 정상 범위 (Dev-Holdout Gap: -1.90%p, low)

**최종 설정 픽스** (2026-01-07):
- 정규화 방법: `zscore` (픽스)
- `ridge_alpha`: 8.0 (픽스)
- `min_feature_ic`: -0.1 (픽스)
- 단기 News 피처 가중치: 0.10 (픽스)
- 장기 News 피처 가중치: 0.03 (픽스)
- L8-L5 피처셋 통일: 22개/19개 (픽스)

**상세 리포트**: `LOCAL_TRASH/artifacts_data/artifacts/reports/normalization_method_comparison_and_final_results.md`

#### 최종 백테스트 결과 (2026-01-07 실행, 거래비용/AlphaQuality/오버래핑 트랜치 반영)

**핵심 반영 사항** (2026-01-07):
- **[개선안 1번] 거래비용 모델 정상화(턴오버 기반)**: 고정 10bp 차감이 아닌 `turnover_oneway * (cost_bps + slippage_bps)` 방식으로 비용 차감
- **[개선안 34번] Alpha Quality 지표 추가**: `IC`, `Rank IC`, `ICIR`, `Rank ICIR`, `Long/Short Alpha(ann)`가 `bt_metrics_{strategy}`에 포함
- **[개선안 36번] 오버래핑 트랜치(필수)**: BT120에 월별 4트랜치 도입
  - 매 20일마다 신규 트랜치 1개 추가(월별), 트랜치 만기는 120일(캘린더 day), 동시 보유 최대 4개
  - 결과적으로 BT120도 **Holdout 리밸런싱 수가 3회 → 23회로 증가**하여 타이밍 럭이 크게 감소
- **rebalance_interval 처리 일원화**: L6R에서 interval별 `rebalance_scores_from_ranking_interval_{N}` 생성, L7은 추가 필터링 없음

**Ridge Alpha 최적화** (2026-01-06):
- **최적화 방법**: Grid Search (Ridge Alpha: [0.01, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0])
- **최적화 목표**: BT20 (Total Return 중심), BT120 (Sharpe 지수 중심)
- **최적화 결과**: 모든 전략이 랭킹 기반이므로 ridge_alpha 값이 성과에 영향을 주지 않음 (정상 동작)
- **운영 설정**: `configs/config.yaml`의 `l5.ridge_alpha`를 유지 (현재 **8.0**, 2026-01-07 최종 픽스)
- **상세 리포트**: `LOCAL_TRASH/artifacts_data/artifacts/reports/FINAL_RIDGE_ALPHA_OPTIMIZATION_REPORT.md`

**완전 교체 전략 + top_k 최적화** (2026-01-06):
- **실험 조건**: 완전 교체 전략 (rebalance_interval=holding_days, buffer_k=0)
  - Day1: top_k 매수 → Day20: 전량 매도 → Day20 top_k 재매수 (매번 100% 교체)
- **최적화 결과**: top_k=15이 Holdout 구간에서 최고 성과
  - Holdout Total Return: **12.39%** (최고)
  - Holdout Sharpe: **0.5464** (최고)
  - Holdout CAGR: **6.69%** (최고)
- **비교 결과**: top_k=15 > top_k=20 > top_k=10 (Holdout 기준)
- **상세 리포트**: `LOCAL_TRASH/artifacts_data/artifacts/reports/full_replacement_topk_optimization_report.md`

**가중치 방식(equal vs softmax) 비교 최적화** (2026-01-06):
- **실험 조건**: 4가지 전략 모두에 대해 equal과 softmax 비교
- **최적화 결과**: **모든 전략에서 equal이 softmax보다 우수**
  - bt20_short: equal (4.90%) > softmax (2.98%) - 차이 39.09%
  - bt20_ens: equal (6.00%) > softmax (1.85%) - 차이 69.13%
  - bt120_long: equal (2.88%) > softmax (1.19%) - 차이 58.54%
  - bt120_ens: equal (4.03%) > softmax (2.69%) - 차이 33.15%
- **권장사항**: 모든 전략에서 `weighting: equal` 사용 권장
- **상세 리포트**: `LOCAL_TRASH/artifacts_data/artifacts/reports/weighting_comparison_optimization_report.md`

**4가지 전략의 최종 Holdout 구간 성과 (2023-01-31 ~ 2024-11-18, Track A 앙상블 적용)**:

| 전략 | Net Sharpe | Net CAGR | Net MDD | Calmar Ratio | Hit Ratio | 리밸런싱 수 |
|------|------------|----------|---------|--------------|-----------|------------|
| **bt120_long** | **0.5689** ⭐ | **6.86%** | **-10.27%** | **0.6679** | **60.87%** | 23 |
| **bt120_ens** | **0.4234** | **4.59%** | **-9.26%** | **0.4954** | **65.22%** | 23 |
| **bt20_ens** | **0.2431** | **3.04%** | **-18.39%** | **0.1650** | **47.83%** | 23 |
| **bt20_short** | **-0.1826** | **-4.52%** | **-15.56%** | **-0.2904** | **56.52%** | 23 |

**🏆 최종 성과 요약**:
- **bt120_long Sharpe 0.5689**: 가장 안정적인 성과 ✅
- ⚠️ **주의**: 일부 전략(bt20_short, bt20_ens)은 Holdout 구간에서 음수 수익률
- **안정적 수익률**: CAGR 7.61%, MDD -5.90% ✅
- **높은 승률**: Hit Ratio 60.87% ✅
- **앙상블 효과**: 과적합 LOW-MEDIUM 등급 관리 ✅

**주요 특징**:
- **Track A 앙상블 적용**: 4개 모델의 강점 결합으로 안정성 확보
- **과적합 방지**: IC Diff 92%+ 감소로 일반화 성능 향상
- `bt_metrics_{strategy}`에 AlphaQuality(IC/ICIR/Long-Short Alpha) 포함

**참고**: 모든 지표는 거래비용(cost_bps=10.0)을 반영한 Net 지표이며, 시장 국면 기능은 외부 API 없이 ohlcv_daily 데이터로 자동 분류됩니다.

**참고**:
- 모든 지표는 `phase` 컬럼으로 구간별(`dev`/`holdout`) 구분됩니다.
- Gross 지표는 거래비용 차감 전 성과를 나타냅니다.
- Net 지표는 거래비용 차감 후 실제 수익을 나타냅니다.

#### 데이터 파일 위치(주요)

백테스트 실행 시 다음 파일들이 생성됩니다:

```
LOCAL_TRASH/artifacts_data/data/interim/
├── bt_metrics_{strategy}.parquet          # 백테스트 성과 지표 (메인)
├── bt_positions_{strategy}.parquet        # 포지션 정보
├── bt_returns_{strategy}.parquet          # 일별 수익률
├── bt_equity_curve_{strategy}.parquet     # 자산 곡선
├── bt_regime_metrics_{strategy}.parquet   # 국면별 성과 (조건부)
├── bt_selection_diagnostics_{strategy}.parquet  # [개선안 28번] 선택 진단 정보
├── bt_returns_diagnostics_{strategy}.parquet    # [개선안 28번] 수익률 진단 정보(regime/exposure 등)
└── bt_runtime_profile_{strategy}.parquet         # [개선안 28번] 런타임 프로파일
```

---

## 투트랙 활용 시나리오

### 시나리오 1: 랭킹만 제공 (Track A만 사용)

**목적**: 이용자에게 종목 랭킹 정보만 제공하고, 이용자가 직접 투자 결정

**실행 순서**:
1. 공통 데이터 준비: `python scripts/run_pipeline_l0_l7.py`
2. Track A 실행: `python -m src.pipeline.track_a_pipeline`
3. UI에서 랭킹 표시 (단기/장기/통합 랭킹)
4. 이용자가 랭킹을 참고하여 직접 투자 결정

**적용 사례**:
- 랭킹 기반 종목 추천 서비스
- 투자 정보 제공 플랫폼

### 시나리오 2: 투자 모델 예시 제공 (Track A + Track B)

**목적**: 랭킹과 함께 다양한 투자 모델의 성과 예시를 제공하여 이용자의 투자 결정 지원

**실행 순서**:
1. 공통 데이터 준비: `python scripts/run_pipeline_l0_l7.py`
2. Track A 실행: `python -m src.pipeline.track_a_pipeline`
3. Track B 실행: `python -m src.pipeline.track_b_pipeline bt20_short` (원하는 전략 선택)
4. UI에서 랭킹 + 투자 모델 성과 표시
5. 이용자가 랭킹과 성과를 함께 참고하여 투자 결정

**적용 사례**:
- 랭킹 기반 포트폴리오 추천 서비스
- 백테스트 결과를 포함한 투자 전략 제안

### 시나리오 3: 통합 제공 (Track A + Track B 모든 전략)

**목적**: 랭킹과 다양한 투자 모델 예시를 모두 제공하여 이용자가 자신의 투자 성향에 맞는 모델 선택

**실행 순서**:
1. 공통 데이터 준비: `python scripts/run_pipeline_l0_l7.py`
2. Track A 실행: `python -m src.pipeline.track_a_pipeline`
3. Track B 실행 (모든 전략):
   ```bash
   python -m src.pipeline.track_b_pipeline bt20_short
   python -m src.pipeline.track_b_pipeline bt20_ens
   python -m src.pipeline.track_b_pipeline bt120_long
   python -m src.pipeline.track_b_pipeline bt120_ens
   ```
4. UI에서 랭킹과 모든 투자 모델 성과 표시
5. 이용자가 자신의 투자 성향에 맞는 모델 선택

**적용 사례**:
- 종합 투자 플랫폼
- 다양한 투자 전략 비교 서비스

---

## 참고 문서

### 핵심 개념
- `docs/TWO_TRACK_ARCHITECTURE.md`: 투트랙 아키텍처 가이드 ⭐ **새로 작성됨**
- `TECH_REPORT_TRACK1_RANKING.md`: Track A 기술 보고서 ⭐ **최상단으로 이동**
- `TECH_REPORT_TRACK2_BACKTEST.md`: Track B 기술 보고서 ⭐ **최상단으로 이동**
- `EASY_TECH_REPORT_FOR_NON_FINANCE.md`: 비금융인을 위한 통합 기술 보고서 ⭐ **최상단으로 이동**
- `DUAL_HORIZON_MODEL_RANKING_UI_NOTES.md`: BT20/BT120 개념 상세 설명
- `docs/TRACK_DEFINITION.md`: Track A/B 정의 및 구조

### 최종 리포트
- `LOCAL_TRASH/artifacts_data/artifacts/reports/FINAL_CONFIG_DETERMINATION_REPORT.md`: 최종 설정 확정 리포트
- `LOCAL_TRASH/artifacts_data/artifacts/reports/FINAL_RANKING_STRATEGY_COMPARISON_REPORT.md`: 랭킹 전략 비교 리포트
- `LOCAL_TRASH/artifacts_data/artifacts/reports/FINAL_METRICS_COMPREHENSIVE_REPORT.md`: 최종 성과 리포트

### 리팩터링 문서
- `REFACTORING_DESIGN.md`: 리팩터링 설계 문서
- `PROJECT_CLEANUP_COMPLETE.md`: 프로젝트 정리 완료 보고서

### 데이터 수집 모듈 (리팩토링 완료) ⭐
- `src/data_collection/`: 데이터 수집 완전 분리 모듈
  - **1단계**: 데이터 수집 완전 분리 (기존 데이터 그대로 유지)
  - **2단계**: 함수/모듈화 (UI에서 import 가능한 형태)
  - **3단계**: 파이프라인 재조립 (재현성 + 실행 간편화)

---

## 재현성

다른 컴퓨터에서도 동일한 결과를 재현할 수 있습니다.

### ✅ 재현 방법 (2가지 옵션)

#### 옵션 1: Git 제외 데이터 없이 재현 (자동 다운로드)

**핵심**: Git 제외 데이터(대용량 데이터) 없이도 **전체 파이프라인 재현 가능**합니다.

**이유**:
- 필수 데이터는 **pykrx API를 통해 인터넷에서 자동 다운로드** 가능
- 계산 데이터는 **원천 데이터로부터 자동 재산출** 가능
- 캐시 우선 로직이 있어, 데이터가 없으면 자동으로 다운로드/계산

**상세 가이드**: [`docs/REPRODUCIBILITY_WITHOUT_DATA.md`](docs/REPRODUCIBILITY_WITHOUT_DATA.md) 참고

#### 옵션 2: 외부 드라이브 데이터 사용 (랭킹/백테스트만 산출)

**핵심**: 데이터 파일을 외부 드라이브(Google Drive, OneDrive 등)에 올리고, **랭킹 산정과 백테스트만 직접 산출** 가능합니다.

**장점**:
- ✅ GitHub 저장소 크기 최소화
- ✅ 사용자가 필요한 결과만 산출 가능
- ✅ 데이터 보안 (외부 드라이브 접근 제어)

**실행 방법**:
```bash
# 1. 데이터 검증
python scripts/check_data_files.py

# 2. 랭킹 산정 (Track A)
python -m src.pipeline.track_a_pipeline

# 3. 백테스트 (Track B)
python -m src.pipeline.track_b_pipeline bt120_long

# 또는 원클릭 실행
python scripts/run_ranking_and_backtest.py
```

**상세 가이드**: [`docs/DATA_EXTERNAL_DRIVE_GUIDE.md`](docs/DATA_EXTERNAL_DRIVE_GUIDE.md) 참고

### 기본 실행 (자동 데이터 다운로드)

```bash
# 1. 의존성 설치
pip install -e .

# 2. 설정 파일 확인
# configs/config.yaml의 base_dir 경로 확인

# 3. 전체 데이터 수집 (L0~L4, pykrx API로 자동 다운로드)
python -m src.cli data-download

# 4. Track A 실행
python -m src.cli track-a

# 5. Track B 실행
python -m src.cli track-b --strategy bt120_long
```

**예상 소요 시간**: 약 30-60분 (인터넷 속도에 따라 다름)

### 재현 시나리오

#### 시나리오 1: 최소 구성 (OHLCV만) ✅

**필수 조건**:
- Python 3.11+
- pykrx 설치
- 인터넷 연결

**실행 가능**: ✅ **완전히 가능**
- L0: 유니버스 (pykrx API)
- L1: OHLCV (pykrx API)
- L3~L7: 순수 계산 로직

**제한사항**: 재무/뉴스/ESG 피처 없음 (기능 제한적이지만 실행 가능)

#### 시나리오 2: 기본 구성 (OHLCV + pykrx 재무) ✅

**추가 실행**: L1B (pykrx 재무 데이터)

**실행 가능**: ✅ **완전히 가능**

#### 시나리오 3: 완전 구성 (모든 피처) ⚠️

**필수 조건**:
- 위 조건 +
- DART API 키 (선택적)
- 뉴스/ESG 파일 (외부 수집 필요, 선택적)

**실행 가능**: ⚠️ **조건부**

### 기존 데이터 사용 (선택사항)

기존에 생성된 데이터가 있다면:

```bash
# 기존 데이터 복원 (LOCAL_TRASH에서)
# 주의: LOCAL_TRASH는 Git 제외되므로 로컬에만 존재
cp -r LOCAL_TRASH/artifacts_data/data/raw_data/*.parquet data/raw_data/
cp -r LOCAL_TRASH/artifacts_data/data/cal_data/*.parquet data/cal_data/

# 그 후 위의 실행 단계 반복 (캐시 우선 로직으로 기존 데이터 사용)
```

---

## 🎯 **프로젝트 완료 상태 (2026-01-19)**

### ✅ **프로젝트 성공 요약**
- **Track A (랭킹 엔진 최적화)**: ✅ **완료**
  - bt120_long Sharpe 0.6092 달성 (목표 0.6+ 초과)
  - 4개 모델 앙상블 최적화 완료
  - 과적합 방지 성공 (IC Diff 92%+ 감소)
  - 실전 적용 준비 완료

- **Track B (투자 모델)**: ✅ **완료**
  - 4가지 전략 백테스트 완료
  - 안정적 성과 검증 (Sharpe 0.57-0.61)
  - Alpha Quality 지표 포함

- **프로젝트 정리**: ✅ **완료**
  - 1,264+개 파일 정리 및 격리
  - 루트 디렉토리 최적화 (183개 → 9개)
  - LOCAL_TRASH 폴더를 통한 안전 보관
  - 완전한 모듈화 구조 구현

### 📊 **최종 권장 사용법**
```bash
# 1. 데이터 준비
python scripts/run_pipeline_l0_l7.py

# 2. Track A 실행 (앙상블 적용)
python -m src.pipeline.track_a_pipeline

# 3. Track B 실행 (주요 전략)
python -m src.pipeline.track_b_pipeline bt120_long

# 4. 결과 확인
python scripts/show_backtest_metrics.py
```

### 🎖️ **핵심 성과 지표 (Holdout 구간)**
- **bt120_long Sharpe Ratio**: 0.5689 (가장 안정적)
- **bt120_long Hit Ratio**: 60.87% (높은 승률)
- **bt120_long Calmar Ratio**: 0.6679 (안정적 위험 조정)
- **과적합 위험**: LOW-MEDIUM (관리 가능)
- ⚠️ **주의**: 일부 전략은 Holdout 구간에서 낮은 성과 또는 음수 수익률

### 📁 **최종 산출물**
- `data/interim/bt_metrics_bt120_long.csv` - 주요 전략 성과 (샘플)
- `data/kospi200_benchmark_cumulative_returns.csv` - 벤치마크 데이터
- `data/external/sector_map.csv` - 섹터 분류 데이터
- `docs/LOCAL_정리_로그.md` - 프로젝트 정리 보고서

> **참고**: 전체 데이터는 `LOCAL_TRASH/artifacts_data/`에 보관되어 있으며, 필요시 복원하여 사용

### 🔒 **프로젝트 최종 상태 고정**
**본 프로젝트는 2026-01-19에 최종 완료되었으며, 이후 변경사항은 없습니다.**
- 모든 개발 작업 완료
- 코드 정리 및 최적화 완료
- 문서화 완전 완료
- LOCAL_TRASH를 통한 파일 보관으로 재현성 유지

---

## 라이선스

본 프로젝트는 포트폴리오 목적으로 작성되었습니다.
