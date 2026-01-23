# GitHub 포트폴리오 정리 작업 - Step 0~2 보고서

**작업 일시**: 2026-01-19  
**작업자**: Tech Lead / Repo Curator  
**목표**: "보기 좋은 구조 + 재현 가능 + 파이프라인 안 깨짐" 3가지를 동시에 만족

---

## Step 0: 현황 스냅샷

### 📂 현재 폴더 트리 (상위 2~3레벨)

```
test-main/
├── .cursor/                    # Cursor 설정
├── .github/                    # CI/CD 워크플로우
│   └── workflows/ci.yml
├── artifacts/                  # 아티팩트 저장소 (현재 .gitkeep만)
├── configs/                    # 설정 파일들 (74개 YAML)
├── data/                       # 데이터 (샘플 + 원본)
│   ├── cal_data/              # 계산된 데이터 (백테스트 결과 등)
│   └── raw_data/              # 원시 데이터
├── data_backup/               # 백업 데이터 (323+ 파일)
├── docs/                      # 문서 (41개 파일)
├── notebooks/                 # Jupyter 노트북
├── reports/                   # 리포트 (10개 파일)
├── scripts/                   # 실행 스크립트 (183개 파일)
├── src/                       # 핵심 소스 코드 (201개 Python)
│   ├── components/
│   ├── core/
│   ├── data_collection/
│   ├── evaluation/
│   ├── features/
│   ├── interfaces/
│   ├── pipeline/
│   ├── portfolio/
│   ├── ranking/
│   ├── scripts/
│   ├── stages/
│   ├── tools/
│   ├── tracks/
│   └── utils/
├── tests/                      # 테스트 코드 (9개 파일)
├── unused/                    # 사용하지 않는 파일 (99개)
├── LICENSE
├── Makefile
├── pyproject.toml
├── pytest.ini
├── README.md
└── .gitignore
```

### 🎯 주요 Entrypoint

1. **전체 파이프라인 (L0~L7)**
   - `scripts/run_pipeline_l0_l7.py` - 레거시 전체 실행
   - `src/data_collection/pipeline.py` - 새로운 데이터 수집 모듈

2. **Track A (랭킹 엔진)**
   - `src/pipeline/track_a_pipeline.py` - 랭킹 생성
   - 실행: `python -m src.pipeline.track_a_pipeline`

3. **Track B (백테스트)**
   - `src/pipeline/track_b_pipeline.py` - 백테스트 실행
   - 실행: `python -m src.pipeline.track_b_pipeline bt120_long`

4. **CLI 인터페이스**
   - `src/cli.py` - 통합 CLI
   - 실행: `python -m src.cli track-a` 또는 `python -m src.cli track-b --strategy bt120_long`

### ⚙️ Configs 위치

- **메인 설정**: `configs/config.yaml`
  - `paths.base_dir`: `C:/Users/seong/OneDrive/Desktop/bootcamp/test-main`
  - 모든 경로는 `{base_dir}` 기준 상대 경로로 관리
- **피처 설정**: `configs/features_*.yaml`, `configs/feature_weights_*.yaml` 등 (74개)

### 🚀 현재 실행 커맨드

```bash
# 1. 데이터 준비 (L0~L4)
python scripts/run_pipeline_l0_l7.py
# 또는
python -m src.cli data-download

# 2. Track A 실행
python -m src.pipeline.track_a_pipeline
# 또는
python -m src.cli track-a

# 3. Track B 실행
python -m src.pipeline.track_b_pipeline bt120_long
# 또는
python -m src.cli track-b --strategy bt120_long
```

### 📊 Git 상태

- Git 저장소 존재 (final 브랜치)
- 일부 삭제된 파일 추적 중 (../000_code/ 경로)
- `.gitignore`에 `LOCAL_TRASH/` 포함됨

---

## Step 1: 안전장치

### ✅ 백업 전략

1. **Git 브랜치 백업** (권장)
   ```bash
   git checkout -b backup/before-portfolio-cleanup
   git add .
   git commit -m "백업: 포트폴리오 정리 전 상태"
   git checkout final  # 원래 브랜치로 복귀
   ```

2. **로컬 백업 폴더** (대안)
   - `_backup_before_portfolio/` 생성 (필요시)

**현재 상태**: Git 저장소가 이미 존재하므로 브랜치 백업 권장

---

## Step 2: 불필요/민감/대용량 파일 분류

### 📋 분류 결과

#### 1️⃣ GitHub 커밋 금지 (데이터, 모델, 캐시, 비밀키)

| 경로/패턴 | 유형 | 크기 추정 | 처리 방법 |
|----------|------|----------|----------|
| `LOCAL_TRASH/` | 대용량 데이터/아티팩트 | 1,264+ 파일 | ✅ 이미 .gitignore에 포함 |
| `data/cal_data/*.parquet` | 백테스트 결과물 | 중간 | ⚠️ .gitignore 추가 필요 |
| `data/raw_data/*.parquet` | 원시 데이터 | 대용량 | ⚠️ .gitignore 추가 필요 |
| `data_backup/` | 백업 데이터 | 대용량 (323+ 파일) | ⚠️ .gitignore 추가 필요 |
| `artifacts/` | 모델/리포트 | 중간 | ✅ 이미 .gitignore에 포함 (artifacts/) |
| `*.pkl`, `*.h5` | 모델 가중치 | 중간 | ⚠️ .gitignore 추가 필요 |
| `.env` | 환경 변수/비밀키 | 소량 | ✅ 이미 .gitignore에 포함 |
| `logs/` | 로그 파일 | 중간 | ✅ 이미 .gitignore에 포함 |
| `__pycache__/`, `*.pyc` | Python 캐시 | 소량 | ✅ 이미 .gitignore에 포함 |
| `.pytest_cache/`, `.mypy_cache/` | 테스트 캐시 | 소량 | ✅ 이미 .gitignore에 포함 |

**민감정보 스캔 결과**:
- ✅ `.env` 파일 없음 (`.env.example`만 존재)
- ✅ `configs/config.yaml`에 `secrets: null` 설정
- ⚠️ `configs/config.yaml`에 하드코딩된 경로 존재 (절대 경로)

#### 2️⃣ 삭제해도 되는 찌꺼기 (캐시, 임시 로그, 빌드 산출물)

| 경로/패턴 | 유형 | 처리 방법 |
|----------|------|----------|
| `unused/` | 사용하지 않는 코드 | ⚠️ 유지 (참고용) 또는 .gitignore |
| `docs/_*.txt` | 분석 임시 파일 | ⚠️ .gitignore 또는 삭제 고려 |
| `*.tmp`, `*.temp` | 임시 파일 | ✅ 이미 .gitignore에 포함 |
| `*.backup`, `*.bak` | 백업 파일 | ✅ 이미 .gitignore에 포함 |
| `configs/*_backup.yaml` | 설정 백업 | ⚠️ .gitignore 추가 고려 |

#### 3️⃣ 커밋 유지 (코드/문서/설정/샘플 소량 데이터)

| 경로/패턴 | 유형 | 비고 |
|----------|------|------|
| `src/` | 핵심 소스 코드 | ✅ 커밋 |
| `tests/` | 테스트 코드 | ✅ 커밋 |
| `scripts/` | 실행 스크립트 | ✅ 커밋 |
| `configs/*.yaml` | 설정 파일 | ✅ 커밋 (백업 제외) |
| `docs/*.md` | 문서 | ✅ 커밋 |
| `data/raw_data/README.md` | 데이터 설명 | ✅ 커밋 |
| `data/cal_data/README.md` | 계산 데이터 설명 | ✅ 커밋 |
| `pyproject.toml`, `Makefile` | 프로젝트 설정 | ✅ 커밋 |
| `LICENSE` | 라이선스 | ✅ 커밋 |

### 🔍 상세 분석

#### 대용량 데이터 위치

1. **LOCAL_TRASH/** (이미 격리됨)
   - `artifacts_data/data/` - 전체 데이터 (interim, external 등)
   - `artifacts_data/artifacts/` - 모델/리포트 파일들
   - `binaries/` - 이미지/PDF 파일들
   - `legacy_experiments/` - 실험 코드들
   - `caches/` - 캐시 파일들

2. **data/** (일부 커밋 가능, 대부분 제외)
   - `data/cal_data/*.parquet` - 백테스트 결과물 (대용량)
   - `data/raw_data/*.parquet` - 원시 데이터 (대용량)
   - `data/raw_data/README.md` - 설명 문서 (커밋 가능)

3. **data_backup/** (전체 제외)
   - 323개 CSV + 323개 Parquet 파일
   - 백업용이므로 GitHub에 불필요

#### 민감정보 확인

- ✅ `.env` 파일 없음
- ✅ `secrets.yaml` 없음
- ⚠️ `configs/config.yaml`에 절대 경로 하드코딩:
  ```yaml
  paths:
    base_dir: C:/Users/seong/OneDrive/Desktop/bootcamp/test-main
  ```
  → `.env.example`에 환경 변수로 대체 권장

---

## 📝 다음 단계 (Step 3~8) 계획

### Step 3: 표준 구조로 리팩터
- 현재 구조가 이미 표준에 가까움
- `artifacts/` 폴더 활용 강화
- `src/cli.py` 개선

### Step 4: 재현성
- `artifacts/runs/<run_id>/manifest.json` 자동 생성
- `metrics.json` 저장 로직 추가

### Step 5: 문서 4종 생성
- `docs/ARCHITECTURE.md` ✅ (이미 존재)
- `docs/STRATEGY_CARD.md` ✅ (이미 존재)
- `docs/DATA_CARD.md` ✅ (이미 존재)
- `docs/MODEL_CARD.md` ✅ (이미 존재)
- → 내용 보완 필요

### Step 6: README.md 포트폴리오 형식
- 현재 README는 상세하지만 포트폴리오 형식으로 재구성 필요

### Step 7: 최소 테스트
- `tests/` 폴더에 스모크 테스트 추가

### Step 8: 최종 점검
- 민감정보 재스캔
- 최종 트리 출력
- 실행 커맨드 검증

---

## ✅ Step 0~2 완료 체크리스트

- [x] 현재 폴더 트리 파악
- [x] 주요 entrypoint 확인
- [x] configs 위치 확인
- [x] 실행 커맨드 확인
- [x] Git 상태 확인
- [x] 파일 분류 (3가지 카테고리)
- [x] 민감정보 스캔
- [x] 대용량 데이터 위치 파악
- [x] .gitignore 현황 확인

**다음 단계**: Step 3 (표준 구조로 리팩터) 진행 준비 완료
