# 외부 드라이브 데이터 사용 가이드

**작성일**: 2026-01-19  
**목적**: 데이터 파일을 외부 드라이브(Google Drive, OneDrive 등)에 올리고, 랭킹/백테스트만 직접 산출하는 방법

---

## ✅ 가능 여부: **완전히 가능**

데이터 파일을 외부 드라이브에 올리고, 랭킹 산정(L8)과 백테스트(L7)만 직접 산출하는 것이 **완전히 가능**합니다.

---

## 📊 데이터 구조

### 외부 드라이브에 올릴 데이터 (L0~L4 산출물)

다음 데이터 파일들을 외부 드라이브에 올립니다:

```
외부_드라이브/data/
├── raw_data/
│   ├── universe_k200_membership_monthly.parquet  # L0 산출물
│   ├── ohlcv_daily.parquet                      # L1 산출물
│   ├── esg_daily.parquet                        # 선택적
│   ├── news_sentiment_daily.parquet             # 선택적
│   └── sector_map.csv                           # 선택적
└── cal_data/
    ├── panel_merged_daily.parquet               # L3 산출물 (필수)
    ├── dataset_daily.parquet                    # L4 산출물 (필수)
    ├── cv_folds_short.parquet                   # L4 산출물 (필수)
    └── cv_folds_long.parquet                    # L4 산출물 (필수)
```

### 사용자가 직접 산출할 데이터 (L8, L6R, L7)

다음 데이터는 사용자가 직접 산출합니다:

```
data/cal_data/
├── ranking_short_daily.parquet      # L8 산출물 (Track A)
├── ranking_long_daily.parquet        # L8 산출물 (Track A)
├── rebalance_scores_from_ranking_interval_*.parquet  # L6R 산출물 (Track B)
├── bt_metrics_*.parquet              # L7 산출물 (Track B)
├── bt_returns_*.parquet              # L7 산출물 (Track B)
├── bt_equity_curve_*.parquet         # L7 산출물 (Track B)
└── bt_positions_*.parquet            # L7 산출물 (Track B)
```

---

## 🔄 실행 방법

### 1단계: 외부 드라이브에서 데이터 다운로드

외부 드라이브(Google Drive, OneDrive 등)에서 데이터를 다운로드하여 프로젝트의 `data/` 폴더에 복사합니다.

```bash
# 예시: Google Drive에서 다운로드한 경우
# data/raw_data/ 폴더에 복사
# data/cal_data/ 폴더에 복사
```

**필수 파일**:
- `data/cal_data/panel_merged_daily.parquet` (또는 `dataset_daily.parquet`)
- `data/cal_data/dataset_daily.parquet`
- `data/cal_data/cv_folds_short.parquet`
- `data/cal_data/cv_folds_long.parquet`
- `data/raw_data/universe_k200_membership_monthly.parquet` (L7용)

**선택적 파일**:
- `data/raw_data/ohlcv_daily.parquet` (시장 국면 분류용, 선택적)
- `data/raw_data/sector_map.csv` (섹터 정보, 선택적)

### 2단계: Track A 실행 (랭킹 산정)

```bash
python -m src.pipeline.track_a_pipeline
```

**입력 요구사항**:
- ✅ `panel_merged_daily.parquet` 또는 `dataset_daily.parquet` (둘 중 하나만 있으면 됨)
- ⚠️ `sector_map.csv` (선택적, 섹터 상대 랭킹 사용 시)

**산출물**:
- `data/cal_data/ranking_short_daily.parquet`
- `data/cal_data/ranking_long_daily.parquet`

### 3단계: Track B 실행 (백테스트)

```bash
python -m src.pipeline.track_b_pipeline bt120_long
```

**입력 요구사항**:
- ✅ `ranking_short_daily.parquet` (Track A 산출물)
- ✅ `ranking_long_daily.parquet` (Track A 산출물)
- ✅ `dataset_daily.parquet` (L4 산출물)
- ✅ `cv_folds_short.parquet` (L4 산출물)
- ✅ `cv_folds_long.parquet` (L4 산출물)
- ✅ `universe_k200_membership_monthly.parquet` (L0 산출물)
- ⚠️ `ohlcv_daily.parquet` (시장 국면 분류용, 선택적)

**산출물**:
- `data/cal_data/rebalance_scores_from_ranking_interval_*.parquet`
- `data/cal_data/bt_metrics_*.parquet`
- `data/cal_data/bt_returns_*.parquet`
- `data/cal_data/bt_equity_curve_*.parquet`
- `data/cal_data/bt_positions_*.parquet`

---

## 📝 상세 실행 가이드

### 시나리오 1: 최소 구성 (랭킹 + 백테스트만)

**외부 드라이브에서 다운로드할 파일**:
```
data/cal_data/
├── panel_merged_daily.parquet       # 필수
├── dataset_daily.parquet            # 필수
├── cv_folds_short.parquet           # 필수
└── cv_folds_long.parquet            # 필수

data/raw_data/
└── universe_k200_membership_monthly.parquet  # 필수 (L7용)
```

**실행 순서**:
```bash
# 1. Track A: 랭킹 산정
python -m src.pipeline.track_a_pipeline

# 2. Track B: 백테스트
python -m src.pipeline.track_b_pipeline bt120_long
```

**예상 소요 시간**: 약 10-20분 (데이터 다운로드 제외)

---

### 시나리오 2: 완전 구성 (시장 국면 분류 포함)

**외부 드라이브에서 다운로드할 파일**:
```
data/cal_data/
├── panel_merged_daily.parquet       # 필수
├── dataset_daily.parquet            # 필수
├── cv_folds_short.parquet           # 필수
└── cv_folds_long.parquet           # 필수

data/raw_data/
├── universe_k200_membership_monthly.parquet  # 필수
├── ohlcv_daily.parquet             # 선택적 (시장 국면 분류용)
└── sector_map.csv                  # 선택적 (섹터 정보)
```

**실행 순서**: 동일

---

## 🛠️ 편의 스크립트 생성

사용자가 쉽게 실행할 수 있도록 편의 스크립트를 제공할 수 있습니다:

### 스크립트: `scripts/run_ranking_and_backtest.py`

```python
"""
랭킹 산정 및 백테스트 실행 스크립트

외부 드라이브에서 데이터를 다운로드한 후 실행합니다.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.track_a_pipeline import run_track_a_pipeline
from src.pipeline.track_b_pipeline import run_track_b_pipeline


def main():
    """랭킹 산정 및 백테스트 실행"""
    config_path = "configs/config.yaml"
    
    print("=" * 80)
    print("랭킹 산정 및 백테스트 실행")
    print("=" * 80)
    
    # 1. Track A: 랭킹 산정
    print("\n[1/2] Track A: 랭킹 산정 실행 중...")
    try:
        result_a = run_track_a_pipeline(
            config_path=config_path,
            force_rebuild=False,  # 캐시 우선
        )
        print("✅ Track A 완료")
    except FileNotFoundError as e:
        print(f"❌ 오류: {e}")
        print("💡 해결: 외부 드라이브에서 다음 파일을 다운로드하세요:")
        print("   - data/cal_data/panel_merged_daily.parquet")
        print("   - data/cal_data/dataset_daily.parquet (선택적)")
        return
    
    # 2. Track B: 백테스트
    print("\n[2/2] Track B: 백테스트 실행 중...")
    strategies = ["bt120_long", "bt20_ens", "bt20_short", "bt120_ens"]
    
    for strategy in strategies:
        print(f"\n  → {strategy} 실행 중...")
        try:
            result_b = run_track_b_pipeline(
                config_path=config_path,
                strategy=strategy,
                force_rebuild=False,  # 캐시 우선
            )
            print(f"  ✅ {strategy} 완료")
        except FileNotFoundError as e:
            print(f"  ⚠️ {strategy} 실패: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("실행 완료!")
    print("=" * 80)
    print("\n산출물 위치:")
    print("  - 랭킹: data/cal_data/ranking_*.parquet")
    print("  - 백테스트: data/cal_data/bt_metrics_*.parquet")


if __name__ == "__main__":
    main()
```

---

## 📋 필수 데이터 체크리스트

### Track A (랭킹 산정) 실행 전 확인

- [ ] `data/cal_data/panel_merged_daily.parquet` 존재
- [ ] 또는 `data/cal_data/dataset_daily.parquet` 존재
- [ ] (선택) `data/raw_data/sector_map.csv` 존재 (섹터 상대 랭킹 사용 시)

### Track B (백테스트) 실행 전 확인

- [ ] `data/cal_data/ranking_short_daily.parquet` 존재 (Track A 산출물)
- [ ] `data/cal_data/ranking_long_daily.parquet` 존재 (Track A 산출물)
- [ ] `data/cal_data/dataset_daily.parquet` 존재
- [ ] `data/cal_data/cv_folds_short.parquet` 존재
- [ ] `data/cal_data/cv_folds_long.parquet` 존재
- [ ] `data/raw_data/universe_k200_membership_monthly.parquet` 존재
- [ ] (선택) `data/raw_data/ohlcv_daily.parquet` 존재 (시장 국면 분류용)

---

## 🔍 데이터 검증 스크립트

데이터 파일이 올바르게 다운로드되었는지 확인하는 스크립트:

```python
"""
데이터 파일 검증 스크립트

외부 드라이브에서 다운로드한 데이터가 올바른지 확인합니다.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_path
from src.utils.io import artifact_exists, load_artifact
import pandas as pd


def check_data_files():
    """필수 데이터 파일 확인"""
    cfg = load_config("configs/config.yaml")
    cal_data_dir = Path(get_path(cfg, "data_cal"))
    raw_data_dir = Path(get_path(cfg, "data_raw_data"))
    
    print("=" * 80)
    print("데이터 파일 검증")
    print("=" * 80)
    
    # Track A 필수 파일
    print("\n[Track A 필수 파일]")
    panel_path = cal_data_dir / "panel_merged_daily"
    dataset_path = cal_data_dir / "dataset_daily"
    
    if artifact_exists(panel_path):
        df = load_artifact(panel_path)
        print(f"  ✅ panel_merged_daily: {len(df):,}행, {len(df.columns)}컬럼")
    elif artifact_exists(dataset_path):
        df = load_artifact(dataset_path)
        print(f"  ✅ dataset_daily: {len(df):,}행, {len(df.columns)}컬럼")
    else:
        print("  ❌ panel_merged_daily 또는 dataset_daily 없음")
        return False
    
    # Track B 필수 파일
    print("\n[Track B 필수 파일]")
    
    required_files = {
        "dataset_daily": cal_data_dir / "dataset_daily",
        "cv_folds_short": cal_data_dir / "cv_folds_short",
        "cv_folds_long": cal_data_dir / "cv_folds_long",
        "universe_k200_membership_monthly": raw_data_dir / "universe_k200_membership_monthly",
    }
    
    all_ok = True
    for name, path in required_files.items():
        if artifact_exists(path):
            df = load_artifact(path)
            print(f"  ✅ {name}: {len(df):,}행")
        else:
            print(f"  ❌ {name} 없음: {path}")
            all_ok = False
    
    # Track A 산출물 (Track B 실행 전 필요)
    print("\n[Track A 산출물]")
    ranking_short_path = cal_data_dir / "ranking_short_daily"
    ranking_long_path = cal_data_dir / "ranking_long_daily"
    
    if artifact_exists(ranking_short_path) and artifact_exists(ranking_long_path):
        short_df = load_artifact(ranking_short_path)
        long_df = load_artifact(ranking_long_path)
        print(f"  ✅ ranking_short_daily: {len(short_df):,}행")
        print(f"  ✅ ranking_long_daily: {len(long_df):,}행")
        print("  → Track B 실행 가능")
    else:
        print("  ⚠️ ranking_*.parquet 없음 (Track A 먼저 실행 필요)")
    
    print("\n" + "=" * 80)
    if all_ok:
        print("✅ 모든 필수 파일 확인 완료")
    else:
        print("❌ 일부 필수 파일이 없습니다. 외부 드라이브에서 다운로드하세요.")
    
    return all_ok


if __name__ == "__main__":
    check_data_files()
```

---

## 📦 외부 드라이브 업로드 가이드

### 업로드할 파일 목록

**필수 파일** (약 500MB~2GB, 데이터 크기에 따라 다름):
```
data/cal_data/
├── panel_merged_daily.parquet       # 필수 (가장 큰 파일)
├── dataset_daily.parquet            # 필수
├── cv_folds_short.parquet           # 필수
└── cv_folds_long.parquet            # 필수

data/raw_data/
└── universe_k200_membership_monthly.parquet  # 필수 (작은 파일)
```

**선택적 파일**:
```
data/raw_data/
├── ohlcv_daily.parquet              # 선택적 (큰 파일)
├── esg_daily.parquet                # 선택적
├── news_sentiment_daily.parquet     # 선택적
└── sector_map.csv                   # 선택적 (작은 파일)
```

### 업로드 방법

1. **Google Drive**:
   - 폴더 압축 후 업로드
   - 또는 직접 폴더 업로드
   - 공유 링크 생성

2. **OneDrive**:
   - 폴더 동기화
   - 또는 직접 업로드
   - 공유 링크 생성

3. **기타 클라우드 스토리지**:
   - Dropbox, Box, AWS S3 등

---

## 🎯 사용자 실행 가이드

### 1. 데이터 다운로드

```bash
# 외부 드라이브에서 데이터 다운로드
# 예: Google Drive에서 다운로드한 경우
# data/ 폴더에 압축 해제
```

### 2. 데이터 검증

```bash
python scripts/check_data_files.py
```

### 3. 랭킹 산정 실행

```bash
python -m src.pipeline.track_a_pipeline
```

### 4. 백테스트 실행

```bash
python -m src.pipeline.track_b_pipeline bt120_long
```

### 5. 결과 확인

```bash
# 백테스트 성과 확인
python scripts/extract_latest_metrics.py
```

---

## ⚠️ 주의사항

### 1. 파일 경로

**설정 파일 확인**:
```yaml
# configs/config.yaml
paths:
  base_dir: <프로젝트 루트 경로>
  data_cal: '{base_dir}/data/cal_data'
  data_raw_data: '{base_dir}/data/raw_data'
```

데이터 파일을 `data/cal_data/`, `data/raw_data/` 폴더에 올바르게 배치해야 합니다.

### 2. 파일 형식

- Parquet 파일: `.parquet` 확장자
- CSV 파일: `.csv` 확장자
- 둘 다 지원 (parquet 우선)

### 3. 데이터 일관성

- 외부 드라이브의 데이터는 **L0~L4까지 완료된 상태**여야 합니다
- 데이터가 불완전하면 오류 발생 가능

### 4. 선택적 데이터

- `ohlcv_daily.parquet`: 시장 국면 분류 기능 사용 시 필요
- `sector_map.csv`: 섹터 상대 랭킹 사용 시 필요
- 없어도 실행 가능 (기능 제한적)

---

## ✅ 최종 결론

**데이터 파일을 외부 드라이브에 올리고, 랭킹/백테스트만 직접 산출하는 것이 완전히 가능합니다.**

**장점**:
- ✅ GitHub 저장소 크기 최소화
- ✅ 사용자가 필요한 결과만 산출 가능
- ✅ 데이터 보안 (외부 드라이브 접근 제어)

**필수 작업**:
1. 외부 드라이브에 L0~L4 산출물 업로드
2. 사용자 가이드 문서 제공
3. 데이터 검증 스크립트 제공
4. 편의 실행 스크립트 제공
