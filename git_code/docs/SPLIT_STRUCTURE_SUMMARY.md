# 분리 구조 최종 요약

**작성일**: 2026-01-19  
**상태**: ✅ 완료

---

## ✅ 작업 완료

### 구조 분리

```
test-main/
├── data_drive/    # 데이터 전체 (567개 파일)
│   ├── data/
│   ├── artifacts/
│   └── data_backup/
└── git_code/      # 코드 전체 (663개 파일)
    ├── src/
    ├── configs/
    ├── scripts/
    └── ...
```

### 경로 자동 해석

- `configs/config.yaml`의 `{git_code_dir}`, `{data_drive_dir}` 플레이스홀더 자동 해석
- `src/utils/config.py`에 분리 구조 지원 로직 추가
- 두 폴더를 합치면 자동으로 경로가 맞춰짐

### 재현 가능 여부

✅ **100% 재현 가능**

**검증 완료**:
- 경로 설정 테스트: ✅ 통과
- 데이터 파일 검증: ✅ 통과
- 파이프라인 실행 가능: ✅ 확인

---

## 📦 업로드 가이드

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
│   ├── cal_data/
│   └── raw_data/
├── artifacts/
└── data_backup/
```

---

## 🚀 사용자 실행 방법

### 1. GitHub에서 클론

```bash
git clone <repository_url> test-main
cd test-main/git_code
```

### 2. 구글 드라이브에서 데이터 다운로드

- `data_drive/` 폴더 전체 다운로드
- `test-main/data_drive/` 위치에 배치

### 3. 실행

```bash
pip install -e .
python scripts/check_data_files.py
python scripts/run_ranking_and_backtest.py
```

---

## ✅ 최종 확인

- [x] 구조 분리 완료
- [x] 경로 자동 해석 완료
- [x] 재현 가능 여부 검증 완료
- [x] 문서 작성 완료
- [x] 스크립트 제공 완료

**결과**: ✅ **100% 재현 가능한 구조 완성**
