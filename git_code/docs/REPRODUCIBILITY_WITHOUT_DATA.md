# Git 제외 데이터 없이 파이프라인 재현 가능 여부

**작성일**: 2026-01-19  
**목적**: GitHub에 올라가지 않는 데이터(대용량 데이터) 없이도 파이프라인을 재현할 수 있는지 확인

---

## ✅ 결론: **재현 가능**

Git 제외 데이터 없이도 **전체 파이프라인 재현 가능**합니다.

**이유**:
- 필수 데이터는 **pykrx API를 통해 인터넷에서 자동 다운로드** 가능
- 계산 데이터는 **원천 데이터로부터 자동 재산출** 가능
- 캐시 우선 로직이 있어, 데이터가 없으면 자동으로 다운로드/계산

---

## 📊 현재 Git에 포함된 데이터

### ✅ Git에 포함된 데이터 (소량 샘플)

```
data/
├── raw_data/
│   ├── README.md          # ✅ Git 포함 (설명 문서)
│   └── sector_map.csv     # ✅ Git 포함 (소량 CSV)
├── cal_data/
│   └── README.md          # ✅ Git 포함 (설명 문서)
└── README.md              # ✅ Git 포함
```

### ❌ Git에 제외된 데이터 (대용량)

```
data/
├── raw_data/
│   ├── universe_k200_membership_monthly.parquet  # ❌ Git 제외
│   ├── ohlcv_daily.parquet                       # ❌ Git 제외
│   ├── esg_daily.parquet                         # ❌ Git 제외
│   └── news_sentiment_daily.parquet              # ❌ Git 제외
└── cal_data/
    ├── panel_merged_daily.parquet                # ❌ Git 제외
    ├── dataset_daily.parquet                     # ❌ Git 제외
    ├── ranking_*.parquet                         # ❌ Git 제외
    └── bt_metrics_*.parquet                      # ❌ Git 제외
```

---

## 🔄 파이프라인 재현 메커니즘

### 1. L0: 유니버스 구성

**필수 데이터**: 없음 (처음부터 생성)

**재현 방법**:
```python
from src.data_collection import collect_universe

# pykrx API를 통해 자동 다운로드
universe = collect_universe(
    start_date="2016-01-01",
    end_date="2024-12-31",
    config_path="configs/config.yaml",
    force_rebuild=True,  # 캐시 무시하고 재다운로드
)
```

**결과**: `data/raw_data/universe_k200_membership_monthly.parquet` 생성

**재현 가능 여부**: ✅ **가능** (pykrx + 인터넷 연결만 필요)

---

### 2. L1: OHLCV 다운로드

**필수 데이터**: `universe_k200_membership_monthly` (L0에서 생성 또는 pykrx로 재다운로드)

**재현 방법**:
```python
from src.data_collection import collect_ohlcv

# pykrx API를 통해 자동 다운로드
ohlcv = collect_ohlcv(
    universe=universe,
    start_date="2016-01-01",
    end_date="2024-12-31",
    config_path="configs/config.yaml",
    force_rebuild=True,
)
```

**결과**: `data/raw_data/ohlcv_daily.parquet` 생성

**재현 가능 여부**: ✅ **가능** (pykrx + 인터넷 연결만 필요)

---

### 3. L3: 패널 병합

**필수 데이터**: `ohlcv_daily` (L1에서 생성)

**선택 데이터**: 
- `fundamentals_annual` (DART, 선택적)
- `esg_daily.parquet` (외부 파일, 선택적)
- `news_sentiment_daily.parquet` (외부 파일, 선택적)

**재현 방법**:
```python
from src.data_collection import collect_panel

# OHLCV만으로도 실행 가능 (재무/ESG/뉴스 없이)
panel = collect_panel(
    ohlcv_daily=ohlcv,
    config_path="configs/config.yaml",
    force_rebuild=True,
)
```

**결과**: `data/cal_data/panel_merged_daily.parquet` 생성

**재현 가능 여부**: ✅ **가능** (OHLCV만 있어도 실행 가능, 선택적 데이터는 스킵)

---

### 4. L4: CV 분할

**필수 데이터**: `panel_merged_daily` (L3에서 생성)

**재현 방법**:
```python
from src.data_collection import collect_dataset

# 순수 계산 로직 (입력만 있으면 재산출 가능)
dataset = collect_dataset(
    panel_merged_daily=panel,
    config_path="configs/config.yaml",
    force_rebuild=True,
)
```

**결과**: 
- `data/cal_data/dataset_daily.parquet`
- `data/cal_data/cv_folds_short.parquet`
- `data/cal_data/cv_folds_long.parquet`

**재현 가능 여부**: ✅ **가능** (순수 계산, 입력만 있으면 재산출 가능)

---

### 5. L8: 랭킹 엔진 (Track A)

**필수 데이터**: `panel_merged_daily` 또는 `dataset_daily` (L3/L4에서 생성)

**재현 방법**:
```python
from src.pipeline.track_a_pipeline import run_track_a_pipeline

# 순수 계산 로직
result = run_track_a_pipeline(
    config_path="configs/config.yaml",
    force_rebuild=True,
)
```

**결과**: 
- `data/cal_data/ranking_short_daily.parquet`
- `data/cal_data/ranking_long_daily.parquet`

**재현 가능 여부**: ✅ **가능** (순수 계산, 입력만 있으면 재산출 가능)

---

### 6. L6R + L7: 백테스트 (Track B)

**필수 데이터**: 
- `ranking_short_daily`, `ranking_long_daily` (Track A에서 생성)
- `dataset_daily` (L4에서 생성)
- `ohlcv_daily` (L1에서 생성)

**재현 방법**:
```python
from src.pipeline.track_b_pipeline import run_track_b_pipeline

# 순수 계산 로직
result = run_track_b_pipeline(
    config_path="configs/config.yaml",
    strategy="bt120_long",
    force_rebuild=True,
)
```

**결과**: 
- `data/cal_data/rebalance_scores_from_ranking_interval_*.parquet`
- `data/cal_data/bt_metrics_*.parquet`
- `data/cal_data/bt_returns_*.parquet`
- `data/cal_data/bt_equity_curve_*.parquet`

**재현 가능 여부**: ✅ **가능** (순수 계산, 입력만 있으면 재산출 가능)

---

## 🎯 재현 시나리오별 요구사항

### 시나리오 1: 최소 구성 (OHLCV만)

**필수 조건**:
- ✅ Python 3.11+
- ✅ pykrx 설치 (`pip install pykrx`)
- ✅ 인터넷 연결

**실행 가능 스테이지**:
- ✅ L0: 유니버스 (pykrx API)
- ✅ L1: OHLCV (pykrx API)
- ✅ L3: 패널 병합 (OHLCV만 사용)
- ✅ L4: CV 분할
- ✅ L8: 랭킹 엔진
- ✅ L6R + L7: 백테스트

**제한사항**:
- ⚠️ 재무 피처 없음 (PER, PBR 등)
- ⚠️ 뉴스/ESG 피처 없음
- ⚠️ 성과는 다를 수 있음 (피처 제한)

**재현 가능 여부**: ✅ **완전히 가능**

---

### 시나리오 2: 기본 구성 (OHLCV + pykrx 재무)

**필수 조건**:
- ✅ Python 3.11+
- ✅ pykrx 설치
- ✅ 인터넷 연결

**추가 실행**:
- ✅ L1B: pykrx 재무 데이터 (PER, PBR, EPS, BPS)

**재현 가능 여부**: ✅ **완전히 가능**

---

### 시나리오 3: 완전 구성 (모든 피처)

**필수 조건**:
- ✅ Python 3.11+
- ✅ pykrx 설치
- ✅ OpenDartReader 설치 (선택적)
- ✅ DART API 키 (선택적)
- ✅ 인터넷 연결
- ⚠️ 뉴스/ESG 파일 (외부 수집 필요, 선택적)

**재현 가능 여부**: ⚠️ **조건부** (API 키 및 외부 파일 필요)

---

## 📝 재현 실행 방법

### 방법 1: 전체 파이프라인 자동 실행

```bash
# 1. 의존성 설치
pip install -e .

# 2. 전체 데이터 수집 (L0~L4)
python -m src.cli data-download

# 3. Track A 실행
python -m src.cli track-a

# 4. Track B 실행
python -m src.cli track-b --strategy bt120_long
```

### 방법 2: 단계별 실행

```python
from src.data_collection import DataCollectionPipeline

# 데이터 수집 파이프라인 생성
pipeline = DataCollectionPipeline(
    config_path="configs/config.yaml",
    force_rebuild=True,  # 캐시 무시하고 재다운로드
)

# L0~L4 실행
pipeline.run_all()

# Track A 실행
from src.pipeline.track_a_pipeline import run_track_a_pipeline
run_track_a_pipeline(force_rebuild=True)

# Track B 실행
from src.pipeline.track_b_pipeline import run_track_b_pipeline
run_track_b_pipeline(strategy="bt120_long", force_rebuild=True)
```

---

## ⚠️ 주의사항

### 1. 다운로드 시간

**예상 소요 시간**:
- L0 (유니버스): ~1분
- L1 (OHLCV): ~10-30분 (KOSPI200 전체, 2016-2024)
- L3 (패널 병합): ~5분
- L4 (CV 분할): ~5분
- L8 (랭킹): ~10분
- L7 (백테스트): ~5분

**총 예상 시간**: 약 **30-60분** (인터넷 속도에 따라 다름)

### 2. API 제한

**pykrx**:
- 공개 API (인증 불필요)
- Rate limiting 없음 (일반적으로)
- 안정적

**DART API** (선택적):
- API 키 필요
- Rate limiting 있음
- 일일 요청 제한 있음

### 3. 데이터 일관성

**주의**: 
- pykrx에서 다운로드한 데이터는 **현재 시점 기준** 최신 데이터
- 기존에 사용한 데이터와 **약간 다를 수 있음** (데이터 업데이트, 수정 등)
- 완전히 동일한 결과를 보장하려면 **기존 데이터 파일 필요**

---

## ✅ 최종 결론

### Git 제외 데이터 없이 재현 가능 여부

| 구성 | 재현 가능 | 필수 조건 | 예상 시간 |
|------|---------|---------|----------|
| **최소 구성** (OHLCV만) | ✅ **가능** | pykrx + 인터넷 | 30-60분 |
| **기본 구성** (OHLCV + pykrx 재무) | ✅ **가능** | pykrx + 인터넷 | 40-70분 |
| **완전 구성** (모든 피처) | ⚠️ **조건부** | 위 조건 + API 키 + 외부 파일 | 50-90분 |

### 핵심 포인트

1. ✅ **필수 데이터는 모두 자동 다운로드 가능**
   - L0: pykrx API
   - L1: pykrx API
   - L3~L7: 순수 계산 로직

2. ✅ **캐시 우선 로직**
   - 데이터가 없으면 자동으로 다운로드/계산
   - `force_rebuild=True`로 강제 재생성 가능

3. ⚠️ **선택적 데이터는 없어도 실행 가능**
   - 재무 데이터: 없어도 실행 가능 (피처 제한)
   - 뉴스/ESG: 없어도 실행 가능 (피처 제한)

4. ⚠️ **결과 일관성**
   - pykrx에서 재다운로드한 데이터는 기존과 약간 다를 수 있음
   - 완전히 동일한 결과를 보장하려면 기존 데이터 파일 필요

---

## 🎯 권장 사항

### GitHub 포트폴리오용

1. **README에 명시**:
   ```markdown
   ## 재현 방법
   
   Git 제외 데이터 없이도 재현 가능합니다:
   - 필수 데이터는 pykrx API를 통해 자동 다운로드
   - 계산 데이터는 원천 데이터로부터 자동 재산출
   - 예상 소요 시간: 30-60분
   ```

2. **최소 샘플 데이터 포함** (선택적):
   - `data/raw_data/sector_map.csv` (이미 포함됨)
   - `data/raw_data/README.md` (이미 포함됨)
   - 소량 샘플 데이터 (1-2개 종목, 1개월)로 빠른 테스트 가능

3. **재현 가이드 문서화**:
   - `docs/REPRODUCE.md`에 상세 가이드 포함
   - 예상 소요 시간 명시
   - 문제 해결 가이드 포함

---

**결론**: Git 제외 데이터 없이도 **전체 파이프라인 재현 가능**합니다. ✅
