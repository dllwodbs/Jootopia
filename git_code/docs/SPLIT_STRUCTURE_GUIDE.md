# 분리 구조 가이드 (data_drive + git_code)

**작성일**: 2026-01-19  
**목적**: data_drive와 git_code로 분리된 구조에서 재현 가능한 파이프라인 실행

---

## 📂 구조 개요

프로젝트는 두 개의 독립적인 폴더로 분리되어 있습니다:

```
test-main/
├── data_drive/          # 데이터 전체 (구글 드라이브용)
│   ├── data/
│   │   ├── cal_data/   # 계산 데이터
│   │   └── raw_data/    # 원시 데이터
│   ├── artifacts/       # 실행 아티팩트
│   └── data_backup/     # 백업 데이터
│
└── git_code/            # 코드 전체 (GitHub용)
    ├── src/             # 소스 코드
    ├── configs/         # 설정 파일
    ├── scripts/         # 실행 스크립트
    ├── tests/           # 테스트 코드
    ├── docs/            # 문서
    └── ...
```

---

## 🎯 사용 시나리오

### 시나리오 1: GitHub에서 클론 + 구글 드라이브에서 데이터 다운로드

1. **GitHub에서 git_code 클론**:
   ```bash
   git clone <repository_url> test-main
   cd test-main/git_code
   ```

2. **구글 드라이브에서 data_drive 다운로드**:
   - **다운로드 링크**: https://drive.google.com/drive/folders/1C-CfAwSpnfJawC6rCejj5QBTMqfDRjiS?usp=drive_link
   - 구글 드라이브에서 `data_drive/` 폴더 전체 다운로드
   - `test-main/data_drive/` 위치에 배치

3. **의존성 설치**:
   ```bash
   cd git_code
   pip install -e .
   ```

4. **파이프라인 실행**:
   ```bash
   # 랭킹 산정
   python -m src.pipeline.track_a_pipeline
   
   # 백테스트
   python -m src.pipeline.track_b_pipeline bt120_long
   ```

---

## 🔧 경로 설정

### 자동 경로 해석

`configs/config.yaml`의 경로는 자동으로 해석됩니다:

```yaml
paths:
  base_dir: '{git_code_dir}'              # git_code 디렉토리
  data_cal: '{data_drive_dir}/data/cal_data'
  data_raw_data: '{data_drive_dir}/data/raw_data'
  data_interim: '{data_drive_dir}/data/cal_data'
  artifacts_models: '{data_drive_dir}/artifacts/models'
  artifacts_reports: '{data_drive_dir}/artifacts/reports'
  logs: '{base_dir}/logs'
```

**플레이스홀더 자동 해석**:
- `{git_code_dir}`: config.yaml이 있는 디렉토리의 부모 (git_code/)
- `{data_drive_dir}`: git_code의 부모의 data_drive/ (../data_drive/)

### 수동 경로 설정 (선택적)

환경 변수로 경로를 지정할 수도 있습니다:

```bash
export BASE_DIR=/path/to/git_code
export DATA_DRIVE_PATH=/path/to/data_drive
```

---

## ✅ 재현 가능 여부

### 완전 재현 가능 ✅

**조건**:
1. ✅ `git_code/` 폴더 (GitHub에서 클론)
2. ✅ `data_drive/` 폴더 (구글 드라이브에서 다운로드)
3. ✅ 두 폴더가 같은 부모 디렉토리(`test-main/`)에 위치

**구조**:
```
test-main/
├── data_drive/    # 구글 드라이브에서 다운로드
└── git_code/      # GitHub에서 클론
```

이 구조에서 **100% 재현 가능**합니다.

---

## 📝 실행 방법

### 1단계: 데이터 확인

```bash
cd git_code
python scripts/check_data_files.py
```

### 2단계: 랭킹 산정

```bash
python -m src.pipeline.track_a_pipeline
```

**필수 데이터**:
- `../data_drive/data/cal_data/panel_merged_daily.parquet`
- 또는 `../data_drive/data/cal_data/dataset_daily.parquet`

### 3단계: 백테스트

```bash
python -m src.pipeline.track_b_pipeline bt120_long
```

**필수 데이터**:
- `../data_drive/data/cal_data/ranking_short_daily.parquet` (Track A 산출물)
- `../data_drive/data/cal_data/ranking_long_daily.parquet` (Track A 산출물)
- `../data_drive/data/cal_data/dataset_daily.parquet`
- `../data_drive/data/cal_data/cv_folds_short.parquet`
- `../data_drive/data/cal_data/cv_folds_long.parquet`
- `../data_drive/data/raw_data/universe_k200_membership_monthly.parquet`

---

## 🔍 경로 확인

### 현재 경로 확인

```python
from src.utils.config import load_config, get_path

cfg = load_config("configs/config.yaml")
print(f"base_dir: {get_path(cfg, 'base_dir')}")
print(f"data_cal: {get_path(cfg, 'data_cal')}")
print(f"data_raw_data: {get_path(cfg, 'data_raw_data')}")
```

### 경로 문제 해결

**문제**: `FileNotFoundError: data_drive/...`

**해결**:
1. `data_drive/` 폴더가 `git_code/`와 같은 부모 디렉토리에 있는지 확인
2. 구조가 다음과 같아야 함:
   ```
   test-main/
   ├── data_drive/
   └── git_code/
   ```

---

## 📦 구글 드라이브

### 다운로드 링크

**data_drive 폴더**: https://drive.google.com/drive/folders/1C-CfAwSpnfJawC6rCejj5QBTMqfDRjiS?usp=drive_link

### 업로드된 폴더

`data_drive/` 폴더 전체가 구글 드라이브에 업로드되어 있습니다.

**포함 내용**:
- `data/cal_data/`: 계산 데이터 (백테스트 결과, 랭킹 등)
- `data/raw_data/`: 원시 데이터 (OHLCV, ESG, 뉴스 등)
- `artifacts/`: 실행 아티팩트 (runs/<run_id>/)
- `data_backup/`: 백업 데이터

**예상 크기**: 약 500MB~2GB (데이터 크기에 따라 다름)

### 공유 설정

- **권한**: 읽기 전용 또는 다운로드 가능
- **링크**: 공유 링크 생성하여 사용자에게 제공

---

## 🎯 최종 구조 요약

### GitHub에 올릴 것 (git_code/)

```
git_code/
├── .github/
├── configs/
├── docs/
├── scripts/
├── src/
├── tests/
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
└── README_PORTFOLIO.md
```

### 구글 드라이브에 올릴 것 (data_drive/)

```
data_drive/
├── data/
│   ├── cal_data/       # 계산 데이터
│   └── raw_data/       # 원시 데이터
├── artifacts/           # 실행 아티팩트
└── data_backup/         # 백업 데이터
```

---

## ✅ 재현 가능 여부 확인

### 체크리스트

- [ ] `git_code/` 폴더 존재 (GitHub에서 클론)
- [ ] `data_drive/` 폴더 존재 (구글 드라이브에서 다운로드)
- [ ] 두 폴더가 같은 부모 디렉토리에 위치
- [ ] `data_drive/data/cal_data/panel_merged_daily.parquet` 존재
- [ ] `data_drive/data/cal_data/dataset_daily.parquet` 존재
- [ ] `data_drive/data/cal_data/cv_folds_short.parquet` 존재
- [ ] `data_drive/data/cal_data/cv_folds_long.parquet` 존재
- [ ] `data_drive/data/raw_data/universe_k200_membership_monthly.parquet` 존재

### 테스트 실행

```bash
cd git_code
python scripts/check_data_files.py
```

모든 체크리스트가 통과하면 **100% 재현 가능**합니다.

---

## 📚 관련 문서

- [`DATA_EXTERNAL_DRIVE_GUIDE.md`](DATA_EXTERNAL_DRIVE_GUIDE.md): 외부 드라이브 데이터 사용 가이드
- [`REPRODUCIBILITY_WITHOUT_DATA.md`](REPRODUCIBILITY_WITHOUT_DATA.md): 데이터 없이 재현 가이드

---

**작성일**: 2026-01-19  
**버전**: v1.0
