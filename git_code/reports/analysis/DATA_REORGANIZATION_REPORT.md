# 데이터 폴더 구조 재구성 리포트

**작업 일시**: 2026-01-22  
**목적**: 데이터를 원천/계산으로 구분하고 폴더 구조 재구성

---

## ✅ 완료된 작업

### 1. 필수/선택 데이터 구분

#### 필수 원천 데이터
- ✅ `universe_k200_membership_monthly`: KOSPI200 멤버십 (L0)
- ✅ `ohlcv_daily`: OHLCV 데이터 (L1)

#### 선택적 원천 데이터
- ⚠️ `fundamentals_annual`: DART 재무 데이터 (L2, API 키 필요)
- ⚠️ `pykrx_fundamentals_daily`: pykrx 재무 데이터 (L1B)
- ⚠️ `esg_daily.parquet`: ESG 데이터
- ⚠️ `news_sentiment_daily.parquet`: 뉴스 감성 데이터
- ⚠️ `sector_map.csv`: 섹터 매핑

#### 필수 계산 데이터
- ✅ `panel_merged_daily`: 패널 병합 (L3)
- ✅ `dataset_daily`: 데이터셋 (L4)
- ✅ `cv_folds_short`: CV 분할 (단기, L4)
- ✅ `cv_folds_long`: CV 분할 (장기, L4)
- ✅ `ranking_short_daily`: 단기 랭킹 (L8)
- ✅ `ranking_long_daily`: 장기 랭킹 (L8)

#### 선택적 계산 데이터
- ⚠️ `rebalance_scores_*`: 리밸런싱 스코어 (L6R)
- ⚠️ `bt_positions_*`: 백테스트 포지션 (L7)
- ⚠️ `bt_returns_*`: 백테스트 수익률 (L7)
- ⚠️ `bt_equity_curve_*`: 백테스트 자산 곡선 (L7)
- ⚠️ `bt_metrics_*`: 백테스트 성과 지표 (L7)

---

### 2. 폴더 구조 재구성

**이전 구조**:
```
data/
├── interim/      # 중간 데이터 (원천 + 계산 혼재)
├── external/     # 외부 데이터
└── processed/    # 처리된 데이터
```

**재구성 후**:
```
data/
├── raw_data/     # 원천 데이터 (다운로드/수집된 원본)
├── cal_data/     # 계산 데이터 (파이프라인 생성)
└── README.md     # 전체 데이터 구조 설명
```

**하위 README 파일**:
- `raw_data/README.md`: 원천 데이터 상세 설명
- `cal_data/README.md`: 계산 데이터 상세 설명

---

### 3. 데이터 이동 결과

**원천 데이터 이동**: 5개
- ✅ `universe_k200_membership_monthly` (필수)
- ✅ `ohlcv_daily` (필수)
- ✅ `esg_daily.parquet` (선택)
- ✅ `news_sentiment_daily.parquet` (선택)
- ✅ `sector_map.csv` (선택)

**계산 데이터 이동**: 48개
- ✅ 필수 계산 데이터: 6개
- ✅ 선택적 계산 데이터: 42개

---

## 📊 최종 데이터 구조

### data/raw_data/

**필수 데이터**:
- `universe_k200_membership_monthly.parquet`
- `ohlcv_daily.parquet`

**선택 데이터**:
- `esg_daily.parquet`
- `news_sentiment_daily.parquet`
- `sector_map.csv`

**설명 파일**:
- `README.md`

---

### data/cal_data/

**필수 데이터**:
- `panel_merged_daily.parquet`
- `dataset_daily.parquet`
- `cv_folds_short.parquet`
- `cv_folds_long.parquet`
- `ranking_short_daily.parquet`
- `ranking_long_daily.parquet`

**선택 데이터**:
- `rebalance_scores_from_ranking_interval_20.*`
- `bt_positions_*` (4개 전략)
- `bt_returns_*` (4개 전략)
- `bt_equity_curve_*` (4개 전략)
- `bt_metrics_*` (4개 전략)
- `bt_returns_diagnostics_*` (4개 전략)

**설명 파일**:
- `README.md`

---

## 📝 생성된 문서

1. **data/README.md**: 전체 데이터 구조 설명
   - 폴더 구조
   - 데이터 분류
   - 파이프라인 데이터 흐름
   - 필수 vs 선택 데이터
   - 재산출 가이드

2. **data/raw_data/README.md**: 원천 데이터 상세 설명
   - 각 원천 데이터 설명
   - 다운로드 방법
   - 필수/선택 구분
   - 재다운로드 가이드

3. **data/cal_data/README.md**: 계산 데이터 상세 설명
   - 각 계산 데이터 설명
   - 생성 단계
   - 재산출 방법
   - 데이터 통계

---

## ✅ 검증 완료

- ✅ 원천 데이터와 계산 데이터 분리 완료
- ✅ 필수/선택 데이터 구분 완료
- ✅ 설명 문서 3개 생성 완료
- ✅ data 폴더 최종 형태: raw_data, cal_data, README.md

---

## 📚 관련 파일

- `scripts/reorganize_data_structure.py`: 재구성 스크립트
- `data/README.md`: 전체 데이터 구조 설명
- `data/raw_data/README.md`: 원천 데이터 설명
- `data/cal_data/README.md`: 계산 데이터 설명

---

**작성자**: AI Assistant  
**작성 일시**: 2026-01-22
