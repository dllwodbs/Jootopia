# 계산 데이터 재산출 리포트

**실행 일시**: 2026-01-22  
**목적**: 원천 데이터는 제외하고 계산 데이터만 재산출

---

## 📊 실행 결과 요약

### ✅ 완료된 작업

1. **외부 데이터 복사** ✅
   - `data_backup`에서 `data/external`로 복사 완료
   - `esg_daily.parquet` ✅
   - `news_sentiment_daily.parquet` ✅
   - `sector_map.csv` ✅

### ⚠️ 문제점

**원천 데이터 부재**:
- `universe_k200_membership_monthly` (L0): 없음
- `ohlcv_daily` (L1): 없음

**영향**: 원천 데이터가 없으면 계산 데이터 재산출 불가능

---

## 🔍 원인 분석

### data_backup 폴더 구조

`data_backup` 폴더에는 다음만 존재:
- ✅ 외부 데이터: `esg_daily.parquet`, `news_sentiment_daily.parquet`, `sector_map.csv`
- ✅ 회사별 데이터: `esg_by_company/`, `news_by_company/`
- ❌ 원천 데이터: `universe_k200_membership_monthly`, `ohlcv_daily` 없음

### 해결 방안

**옵션 1: 원천 데이터 다운로드** (권장)
- pykrx를 사용하여 인터넷에서 다운로드
- 시간이 오래 걸릴 수 있음 (전체 기간: 2016-2024)
- 인터넷 연결 필요

**옵션 2: 기존 원천 데이터 사용**
- 다른 위치에 원천 데이터가 있다면 `data/interim`으로 복사
- 또는 `data_backup`에 원천 데이터 추가

---

## 📋 재산출 스크립트 상태

**스크립트**: `scripts/recompute_calculated_data.py`

**기능**:
1. ✅ `data_backup`의 외부 데이터를 `data/external`로 복사
2. ⚠️ 원천 데이터 확인 및 다운로드 (실패)
3. ⏸️ L3 패널 병합 재산출 (원천 데이터 필요)
4. ⏸️ L4 CV 분할 재산출 (L3 필요)
5. ⏸️ L8 랭킹 엔진 재산출 (L3/L4 필요)
6. ⏸️ L6R, L7 백테스트 재산출 (Track A 필요)

---

## 🎯 다음 단계

### 즉시 조치 필요

1. **원천 데이터 준비**
   ```bash
   # 옵션 1: 다운로드 (인터넷 연결 필요)
   python -m src.data_collection.pipeline
   
   # 옵션 2: 기존 데이터 복사
   # universe_k200_membership_monthly.parquet를 data/interim/로 복사
   # ohlcv_daily.parquet를 data/interim/로 복사
   ```

2. **재산출 재시도**
   ```bash
   python scripts/recompute_calculated_data.py
   ```

---

## 📝 결론

**현재 상태**: 원천 데이터 부재로 인해 계산 데이터 재산출 불가능

**완료된 작업**:
- ✅ 외부 데이터 복사 완료
- ✅ 스크립트 작성 완료

**필요한 작업**:
- ⚠️ 원천 데이터 준비 (다운로드 또는 복사)
- ⏸️ 계산 데이터 재산출 (원천 데이터 준비 후)

---

**리포트 작성자**: AI Assistant  
**작성 일시**: 2026-01-22
