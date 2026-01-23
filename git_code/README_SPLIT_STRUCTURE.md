# 분리 구조 사용 가이드

**⚠️ 중요**: 이 프로젝트는 **분리 구조**로 구성되어 있습니다.

---

## 📂 구조

```
test-main/
├── data_drive/    # 데이터 전체 (구글 드라이브용)
│   ├── data/
│   └── artifacts/
└── git_code/      # 코드 전체 (GitHub용, 현재 폴더)
    ├── src/
    ├── configs/
    └── ...
```

---

## 🚀 빠른 시작

### 1. GitHub에서 git_code 클론

```bash
git clone <repository_url> test-main
cd test-main/git_code
```

### 2. 구글 드라이브에서 data_drive 다운로드

**다운로드 링크**: https://drive.google.com/drive/folders/1C-CfAwSpnfJawC6rCejj5QBTMqfDRjiS?usp=drive_link

- 구글 드라이브에서 `data_drive/` 폴더 전체 다운로드
- `test-main/data_drive/` 위치에 배치

**구조 확인**:
```
test-main/
├── data_drive/    # 구글 드라이브에서 다운로드
└── git_code/      # GitHub에서 클론 (현재 위치)
```

### 3. 의존성 설치

```bash
pip install -e .
```

### 4. 데이터 검증

```bash
python scripts/check_data_files.py
```

### 5. 파이프라인 실행

```bash
# 랭킹 산정
python -m src.pipeline.track_a_pipeline

# 백테스트
python -m src.pipeline.track_b_pipeline bt120_long
```

---

## 📚 상세 가이드

- [`docs/SPLIT_STRUCTURE_GUIDE.md`](docs/SPLIT_STRUCTURE_GUIDE.md): 분리 구조 상세 가이드
- [`docs/DATA_EXTERNAL_DRIVE_GUIDE.md`](docs/DATA_EXTERNAL_DRIVE_GUIDE.md): 외부 드라이브 데이터 사용 가이드

---

**작성일**: 2026-01-19
