# test-main 파이프라인 실행 가능 여부 점검 리포트

**점검 일시**: 2026-01-20  
**점검 범위**: 전체 파이프라인 실행 가능 여부 (원천 데이터 사용, 계산 데이터 재산출 가능 여부)

---

## 📊 점검 개요

- **기준 폴더**: `test-main`
- **점검 목적**: 원천 데이터는 그대로 사용, 계산 데이터는 새로 산출 가능한지 확인
- **점검 범위**: L0 → L4 → Track A (L8) → Track B (L6R, L7)

---

## 🔍 스테이지별 실행 가능 여부 점검

### ✅ L0: 유니버스 구성

**목적**: KOSPI200 멤버십 정보 수집

**원천 데이터**:
- ✅ **pykrx 라이브러리 사용**: 인터넷에서 실시간 다운로드 가능
- ✅ **인터넷 연결 필요**: KRX 웹사이트에서 데이터 수집
- ✅ **API 키 불필요**: pykrx는 공개 API 사용

**실행 가능 여부**: ✅ **가능**
- `src/tracks/shared/stages/data/l0_universe.py` 존재
- `build_k200_membership_monthly()` 함수 구현됨
- pykrx 의존성: `pyproject.toml`에 포함됨

**산출물**: `universe_k200_membership_monthly.parquet`
- 날짜별 KOSPI200 구성 종목 리스트
- **재산출 가능**: ✅ 가능 (pykrx로 매번 다운로드)

---

### ✅ L1: OHLCV 데이터 + 기술적 지표

**목적**: 일별 주가 데이터 및 기술적 지표 계산

**원천 데이터**:
- ✅ **pykrx 라이브러리 사용**: 인터넷에서 실시간 다운로드 가능
- ✅ **인터넷 연결 필요**: KRX 웹사이트에서 데이터 수집
- ✅ **API 키 불필요**: pykrx는 공개 API 사용

**계산 데이터**:
- ✅ **기술적 지표 자동 계산**: RSI, MACD, 볼린저 밴드 등
- ✅ **코드에 구현됨**: `src/tracks/shared/stages/data/l1_ohlcv.py`

**실행 가능 여부**: ✅ **가능**
- `download_ohlcv_panel()` 함수 구현됨
- 기술적 지표 계산 로직 포함
- pykrx 의존성 확인됨

**산출물**: `ohlcv_daily.parquet`
- 일별 OHLCV + 기술적 지표
- **재산출 가능**: ✅ 가능 (pykrx로 다운로드 + 계산)

---

### ⚠️ L2: 재무 데이터 (DART)

**목적**: 공시 기반 재무 지표 수집

**원천 데이터**:
- ⚠️ **OpenDartReader 사용**: DART API 사용
- ⚠️ **API 키 필요**: DART API 키 등록 필요
- ⚠️ **인터넷 연결 필요**: DART 웹사이트에서 데이터 수집

**실행 가능 여부**: ⚠️ **조건부 가능**
- `src/tracks/shared/stages/data/l2_fundamentals_dart.py` 존재
- `download_annual_fundamentals()` 함수 구현됨
- **필수 조건**: DART API 키 필요

**산출물**: `fundamentals_annual.parquet`
- 연간 재무 지표 (매출, 영업이익, 당기순이익 등)
- **재산출 가능**: ⚠️ 조건부 (API 키 필요)

**대안**:
- ✅ **pykrx 재무 데이터 (L1B)**: PER, PBR, EPS, BPS 등
- ✅ **API 키 불필요**: pykrx로 일부 재무 지표 수집 가능
- ✅ **선택적 사용**: `src/tracks/shared/stages/data/l1b_pykrx_fundamentals.py`

---

### ✅ L3: 패널 병합

**목적**: OHLCV + 재무 + 뉴스 + ESG 데이터 병합

**입력 데이터**:
- ✅ `ohlcv_daily`: L1 산출물 (재산출 가능)
- ⚠️ `fundamentals_annual`: L2 산출물 (API 키 필요)
- ⚠️ `pykrx_fundamentals_daily`: L1B 산출물 (선택적, 재산출 가능)
- ⚠️ `news_sentiment_daily`: 외부 파일 (`data/external/news_sentiment_daily.parquet`)
- ⚠️ `esg_daily`: 외부 파일 (`data/external/esg_daily.parquet`)

**실행 가능 여부**: ✅ **가능** (선택적 데이터는 스킵 가능)
- `src/tracks/shared/stages/data/l3_panel_merge.py` 존재
- `build_panel_merged_daily()` 함수 구현됨
- **핵심 정책**: 외부 데이터(뉴스/ESG)가 없어도 실패하지 않고 스킵

**산출물**: `panel_merged_daily.parquet`
- 모든 피처가 병합된 패널 데이터
- **재산출 가능**: ✅ 가능 (입력 데이터만 있으면 재산출)

**의존성**:
- ✅ 필수: `ohlcv_daily` (재산출 가능)
- ⚠️ 선택: `fundamentals_annual` (API 키 필요, 또는 pykrx 대체)
- ⚠️ 선택: `news_sentiment_daily` (외부 파일, 없으면 스킵)
- ⚠️ 선택: `esg_daily` (외부 파일, 없으면 스킵)

---

### ✅ L4: Walk-Forward CV 분할

**목적**: 학습/검증/홀드아웃 구간 분할 및 타겟 생성

**입력 데이터**:
- ✅ `panel_merged_daily`: L3 산출물 (재산출 가능)

**계산 데이터**:
- ✅ **CV 분할 로직**: 코드에 구현됨
- ✅ **타겟 생성**: forward return 계산

**실행 가능 여부**: ✅ **가능**
- `src/tracks/shared/stages/data/l4_walkforward_split.py` 존재
- `build_targets_and_folds()` 함수 구현됨
- **순수 계산**: 입력 데이터만 있으면 재산출 가능

**산출물**:
- `dataset_daily.parquet`: 타겟이 추가된 데이터셋
- `cv_folds_short.parquet`: 단기 CV 분할 정보
- `cv_folds_long.parquet`: 장기 CV 분할 정보

**재산출 가능**: ✅ **완전히 가능** (순수 계산)

---

### ✅ Track A: L8 랭킹 엔진

**목적**: 피처 기반 종목 랭킹 생성

**입력 데이터**:
- ✅ `panel_merged_daily`: L3 산출물 (재산출 가능)
- ✅ `dataset_daily`: L4 산출물 (재산출 가능)

**계산 데이터**:
- ✅ **랭킹 계산 로직**: 코드에 구현됨
- ✅ **피처 가중치**: 설정 파일에서 로드
- ✅ **정규화**: zscore 등 구현됨

**실행 가능 여부**: ✅ **가능**
- `src/tracks/track_a/stages/ranking/l8_dual_horizon.py` 존재
- `run_L8_short_rank_engine()`, `run_L8_long_rank_engine()` 함수 구현됨
- **순수 계산**: 입력 데이터만 있으면 재산출 가능

**산출물**:
- `ranking_short_daily.parquet`: 단기 랭킹
- `ranking_long_daily.parquet`: 장기 랭킹

**재산출 가능**: ✅ **완전히 가능** (순수 계산)

---

### ✅ Track B: L6R 랭킹 스코어 변환

**목적**: 랭킹 데이터를 백테스트용 스코어로 변환

**입력 데이터**:
- ✅ `ranking_short_daily`: Track A 산출물 (재산출 가능)
- ✅ `ranking_long_daily`: Track A 산출물 (재산출 가능)
- ✅ `dataset_daily`: L4 산출물 (재산출 가능)
- ✅ `cv_folds_short`: L4 산출물 (재산출 가능)

**계산 데이터**:
- ✅ **스코어 변환 로직**: 코드에 구현됨
- ✅ **단기/장기 결합**: α 가중치로 결합

**실행 가능 여부**: ✅ **가능**
- `src/tracks/track_b/stages/modeling/l6r_ranking_scoring.py` 존재
- `run_L6R_ranking_scoring()` 함수 구현됨
- **순수 계산**: 입력 데이터만 있으면 재산출 가능

**산출물**: `rebalance_scores_from_ranking_interval_{N}.parquet`
- 백테스트용 스코어 데이터

**재산출 가능**: ✅ **완전히 가능** (순수 계산)

---

### ✅ Track B: L7 백테스트

**목적**: 백테스트 실행 및 성과 지표 계산

**입력 데이터**:
- ✅ `rebalance_scores`: L6R 산출물 (재산출 가능)
- ✅ `ohlcv_daily`: L1 산출물 (재산출 가능, 시장 국면 분류용)

**계산 데이터**:
- ✅ **백테스트 로직**: 코드에 구현됨
- ✅ **성과 지표 계산**: Sharpe, CAGR, MDD 등
- ✅ **시장 국면 분류**: ohlcv_daily 기반 자동 분류 (외부 API 불필요)

**실행 가능 여부**: ✅ **가능**
- `src/tracks/track_b/stages/backtest/l7_backtest.py` 존재
- `run_backtest()` 함수 구현됨
- **순수 계산**: 입력 데이터만 있으면 재산출 가능

**산출물**:
- `bt_positions_{strategy}.parquet`: 포지션 정보
- `bt_returns_{strategy}.parquet`: 일별 수익률
- `bt_equity_curve_{strategy}.parquet`: 자산 곡선
- `bt_metrics_{strategy}.parquet`: 성과 지표

**재산출 가능**: ✅ **완전히 가능** (순수 계산)

---

## 📋 원천 데이터 요약

### ✅ 인터넷에서 다운로드 가능 (API 키 불필요)

1. **L0: 유니버스**
   - pykrx → KRX 웹사이트
   - ✅ 재산출 가능

2. **L1: OHLCV**
   - pykrx → KRX 웹사이트
   - ✅ 재산출 가능

3. **L1B: pykrx 재무 데이터** (선택적)
   - pykrx → PER, PBR, EPS, BPS 등
   - ✅ 재산출 가능
   - ⚠️ L2 (DART)보다 제한적이지만 API 키 불필요

### ⚠️ API 키 필요

1. **L2: DART 재무 데이터**
   - OpenDartReader → DART API
   - ⚠️ API 키 필요
   - ⚠️ 재산출 가능 (API 키 있으면)

### ⚠️ 외부 파일 필요 (선택적)

1. **뉴스 감성 데이터**
   - `data/external/news_sentiment_daily.parquet`
   - ⚠️ 파일 없으면 스킵 (실패하지 않음)
   - ⚠️ 재산출 불가능 (별도 수집 필요)

2. **ESG 데이터**
   - `data/external/esg_daily.parquet`
   - ⚠️ 파일 없으면 스킵 (실패하지 않음)
   - ⚠️ 재산출 불가능 (별도 수집 필요)

---

## 📊 계산 데이터 재산출 가능 여부

| 스테이지 | 입력 데이터 | 계산 로직 | 재산출 가능 | 비고 |
|---------|-----------|---------|-----------|------|
| **L0** | 인터넷 (pykrx) | 멤버십 추출 | ✅ 가능 | 매번 다운로드 |
| **L1** | 인터넷 (pykrx) | OHLCV + 기술적 지표 | ✅ 가능 | 매번 다운로드 + 계산 |
| **L1B** | 인터넷 (pykrx) | 재무 지표 추출 | ✅ 가능 | 매번 다운로드 |
| **L2** | DART API | 재무 지표 추출 | ⚠️ 조건부 | API 키 필요 |
| **L3** | L1, L2, L1B 산출물 | 데이터 병합 | ✅ 가능 | 입력만 있으면 재산출 |
| **L4** | L3 산출물 | CV 분할 + 타겟 생성 | ✅ 가능 | 순수 계산 |
| **L8** | L3/L4 산출물 | 랭킹 계산 | ✅ 가능 | 순수 계산 |
| **L6R** | Track A 산출물 | 스코어 변환 | ✅ 가능 | 순수 계산 |
| **L7** | L6R 산출물 | 백테스트 | ✅ 가능 | 순수 계산 |

---

## ⚠️ 발견된 문제점

### 1. 설정 파일 경로 불일치

**문제**:
- `configs/config.yaml`의 `paths.base_dir`이 `000_code`로 설정됨
- 실제 폴더는 `test-main`

**영향**: 설정 파일 로드 시 오류 발생

**해결책**:
```yaml
paths:
  base_dir: C:/Users/seong/OneDrive/Desktop/bootcamp/test-main
```

### 2. DART API 키 필요

**문제**:
- L2 스테이지에서 OpenDartReader 사용 시 API 키 필요
- API 키 없으면 재무 데이터 수집 불가능

**영향**: 
- L2 스테이지 실행 불가능
- L3에서 재무 데이터 없이 병합 (경고만 발생)

**해결책**:
- DART API 키 등록: https://opendart.fss.or.kr/
- 또는 L1B (pykrx 재무 데이터) 사용 (제한적이지만 API 키 불필요)

### 3. 외부 데이터 파일 부재

**문제**:
- `data/external/news_sentiment_daily.parquet` 없음
- `data/external/esg_daily.parquet` 없음

**영향**:
- 뉴스/ESG 피처 사용 불가능
- ⚠️ **하지만 실패하지 않음**: 코드에서 스킵 처리

**해결책**:
- 외부 데이터 파일 제공 또는
- 해당 피처 없이 실행 (기능 제한적이지만 실행 가능)

---

## ✅ 실행 가능 여부 종합 평가

### 전체 파이프라인 실행 가능 여부

| 시나리오 | 실행 가능 | 조건 |
|---------|---------|------|
| **최소 구성** (OHLCV만) | ✅ 가능 | pykrx 설치 + 인터넷 연결 |
| **기본 구성** (OHLCV + pykrx 재무) | ✅ 가능 | pykrx 설치 + 인터넷 연결 |
| **완전 구성** (OHLCV + DART 재무) | ⚠️ 조건부 | pykrx + OpenDartReader + API 키 |
| **최적 구성** (모든 피처) | ⚠️ 조건부 | 위 조건 + 뉴스/ESG 파일 |

### 스테이지별 실행 가능 여부

| 스테이지 | 필수 조건 | 선택 조건 | 실행 가능 |
|---------|---------|---------|---------|
| **L0** | pykrx + 인터넷 | - | ✅ 가능 |
| **L1** | pykrx + 인터넷 | - | ✅ 가능 |
| **L1B** | pykrx + 인터넷 | - | ✅ 가능 (선택적) |
| **L2** | OpenDartReader + API 키 | - | ⚠️ 조건부 |
| **L3** | L1 산출물 | L2, 뉴스, ESG | ✅ 가능 (선택적 데이터 스킵) |
| **L4** | L3 산출물 | - | ✅ 가능 |
| **L8** | L3/L4 산출물 | - | ✅ 가능 |
| **L6R** | Track A 산출물 | - | ✅ 가능 |
| **L7** | L6R 산출물 | - | ✅ 가능 |

---

## 🎯 권장 실행 시나리오

### 시나리오 1: 최소 구성 (OHLCV만)

**조건**:
- ✅ pykrx 설치
- ✅ 인터넷 연결

**실행 가능 스테이지**:
- ✅ L0: 유니버스
- ✅ L1: OHLCV + 기술적 지표
- ✅ L3: 패널 병합 (재무 데이터 없이)
- ✅ L4: CV 분할
- ✅ Track A (L8): 랭킹 생성
- ✅ Track B (L6R, L7): 백테스트

**제한사항**:
- ⚠️ 재무 피처 없음 (PER, PBR 등)
- ⚠️ 뉴스/ESG 피처 없음

**실행 가능 여부**: ✅ **가능**

### 시나리오 2: 기본 구성 (OHLCV + pykrx 재무)

**조건**:
- ✅ pykrx 설치
- ✅ 인터넷 연결

**실행 가능 스테이지**:
- ✅ L0: 유니버스
- ✅ L1: OHLCV + 기술적 지표
- ✅ L1B: pykrx 재무 데이터 (PER, PBR, EPS, BPS)
- ✅ L3: 패널 병합 (pykrx 재무 포함)
- ✅ L4: CV 분할
- ✅ Track A (L8): 랭킹 생성
- ✅ Track B (L6R, L7): 백테스트

**제한사항**:
- ⚠️ DART 재무 데이터 없음 (더 상세한 재무 지표)
- ⚠️ 뉴스/ESG 피처 없음

**실행 가능 여부**: ✅ **가능**

### 시나리오 3: 완전 구성 (모든 데이터)

**조건**:
- ✅ pykrx 설치
- ✅ OpenDartReader 설치
- ✅ DART API 키
- ✅ 인터넷 연결
- ⚠️ 뉴스/ESG 파일 (선택적)

**실행 가능 스테이지**:
- ✅ 모든 스테이지

**실행 가능 여부**: ⚠️ **조건부** (API 키 필요)

---

## 📝 계산 데이터 재산출 가능 여부 상세

### ✅ 완전히 재산출 가능 (순수 계산)

1. **L4: CV 분할**
   - 입력: `panel_merged_daily`
   - 계산: Walk-Forward CV 분할 로직
   - ✅ **재산출 가능**: 입력만 있으면 매번 재산출 가능

2. **L8: 랭킹 엔진**
   - 입력: `panel_merged_daily` 또는 `dataset_daily`
   - 계산: 피처 가중치 적용, 정규화, 랭킹 계산
   - ✅ **재산출 가능**: 입력만 있으면 매번 재산출 가능

3. **L6R: 랭킹 스코어 변환**
   - 입력: `ranking_short_daily`, `ranking_long_daily`
   - 계산: 단기/장기 결합, 스코어 변환
   - ✅ **재산출 가능**: 입력만 있으면 매번 재산출 가능

4. **L7: 백테스트**
   - 입력: `rebalance_scores`, `ohlcv_daily`
   - 계산: 포지션 생성, 수익률 계산, 성과 지표 계산
   - ✅ **재산출 가능**: 입력만 있으면 매번 재산출 가능

### ⚠️ 조건부 재산출 가능

1. **L0, L1, L1B: 원천 데이터 다운로드**
   - ✅ **재산출 가능**: 인터넷 연결만 있으면 매번 다운로드 가능
   - ⚠️ **주의**: 데이터 소스(예: KRX)의 가용성에 의존

2. **L2: DART 재무 데이터**
   - ⚠️ **조건부**: API 키 필요
   - ✅ **재산출 가능**: API 키 있으면 매번 다운로드 가능

### ❌ 재산출 불가능 (외부 파일 필요)

1. **뉴스 감성 데이터**
   - ❌ **재산출 불가능**: 별도 수집 파이프라인 필요
   - ⚠️ **선택적**: 파일 없어도 파이프라인 실행 가능 (스킵)

2. **ESG 데이터**
   - ❌ **재산출 불가능**: 별도 수집 파이프라인 필요
   - ⚠️ **선택적**: 파일 없어도 파이프라인 실행 가능 (스킵)

---

## 🔧 실행 전 필수 조치사항

### 1. 설정 파일 수정

**문제**: `configs/config.yaml`의 `base_dir` 경로 불일치

**해결책**:
```yaml
paths:
  base_dir: C:/Users/seong/OneDrive/Desktop/bootcamp/test-main
```

### 2. 의존성 설치

**필수 패키지**:
```bash
pip install -e .
# 또는
pip install pykrx pandas numpy scikit-learn xgboost pyarrow
```

**선택 패키지** (L2 사용 시):
```bash
pip install OpenDartReader
```

### 3. DART API 키 설정 (선택적)

**방법**:
- DART 웹사이트에서 API 키 발급: https://opendart.fss.or.kr/
- 환경 변수 또는 설정 파일에 API 키 설정

---

## 📊 실행 가능 여부 최종 평가

### ✅ 핵심 파이프라인 실행 가능

**결론**: **전체 파이프라인 실행 가능** ✅

**조건**:
1. ✅ pykrx 설치
2. ✅ 인터넷 연결
3. ⚠️ DART API 키 (선택적, L1B로 대체 가능)
4. ⚠️ 뉴스/ESG 파일 (선택적, 없어도 실행 가능)

### 실행 가능 스테이지

| 스테이지 | 실행 가능 | 비고 |
|---------|---------|------|
| **L0** | ✅ | pykrx + 인터넷 |
| **L1** | ✅ | pykrx + 인터넷 |
| **L1B** | ✅ | pykrx + 인터넷 (선택적) |
| **L2** | ⚠️ | API 키 필요 (선택적) |
| **L3** | ✅ | L1 산출물만 있어도 가능 |
| **L4** | ✅ | L3 산출물만 있으면 가능 |
| **L8** | ✅ | L3/L4 산출물만 있으면 가능 |
| **L6R** | ✅ | Track A 산출물만 있으면 가능 |
| **L7** | ✅ | L6R 산출물만 있으면 가능 |

### 계산 데이터 재산출 가능 여부

**결론**: **모든 계산 데이터 재산출 가능** ✅

- ✅ L4, L8, L6R, L7: 순수 계산 로직, 입력만 있으면 재산출 가능
- ✅ L0, L1, L1B: 인터넷에서 매번 다운로드 가능
- ⚠️ L2: API 키 있으면 재산출 가능
- ❌ 뉴스/ESG: 별도 수집 필요 (선택적)

---

## 🎯 최종 권장사항

### 즉시 실행 가능한 구성

1. **최소 구성 실행**:
   ```bash
   # 1. 설정 파일 수정 (base_dir)
   # 2. 의존성 설치
   pip install -e .
   
   # 3. 데이터 수집 (L0~L4)
   python -m src.data_collection.collect_all_data
   
   # 4. Track A 실행
   python -m src.pipeline.track_a_pipeline
   
   # 5. Track B 실행
   python -m src.pipeline.track_b_pipeline bt120_long
   ```

2. **기본 구성 실행** (L1B 포함):
   - 위와 동일하되 L1B (pykrx 재무) 활성화
   - 재무 피처 일부 사용 가능

### 개선 권장사항

1. **DART API 키 등록** (선택적)
   - 더 상세한 재무 데이터 수집 가능
   - L2 스테이지 활성화

2. **외부 데이터 파일 제공** (선택적)
   - 뉴스/ESG 피처 사용 가능
   - 더 풍부한 피처셋

---

## ✅ 점검 완료 항목

- [x] L0 실행 가능 여부 확인
- [x] L1 실행 가능 여부 확인
- [x] L2 실행 가능 여부 확인
- [x] L3 실행 가능 여부 확인
- [x] L4 실행 가능 여부 확인
- [x] Track A (L8) 실행 가능 여부 확인
- [x] Track B (L6R, L7) 실행 가능 여부 확인
- [x] 원천 데이터 재산출 가능 여부 확인
- [x] 계산 데이터 재산출 가능 여부 확인
- [x] 의존성 확인
- [x] 설정 파일 확인

---

## 📝 결론

**test-main 폴더는 전체 파이프라인 실행이 가능합니다** ✅

**주요 특징**:
- ✅ **원천 데이터**: pykrx로 인터넷에서 다운로드 가능
- ✅ **계산 데이터**: 모든 중간 산출물 재산출 가능
- ✅ **선택적 데이터**: 뉴스/ESG 없어도 실행 가능 (스킵 처리)
- ⚠️ **조건부**: DART API 키는 선택적 (L1B로 대체 가능)

**실행 가능 여부**: ✅ **가능** (최소 구성 기준)

**재산출 가능 여부**: ✅ **가능** (모든 계산 데이터)

---

**점검자**: AI Assistant  
**점검 일시**: 2026-01-20
