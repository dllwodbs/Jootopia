"""
test-main을 data_drive/와 git_code/로 분리하는 스크립트

구조:
test-main/
├── data_drive/          # 데이터 전체 (구글 드라이브용)
│   ├── data/
│   └── artifacts/
└── git_code/            # 코드 전체 (GitHub용)
    ├── src/
    ├── configs/
    ├── scripts/
    └── ...
"""

import shutil
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent


def create_structure():
    """data_drive/와 git_code/ 구조 생성"""
    
    data_drive_dir = project_root / "data_drive"
    git_code_dir = project_root / "git_code"
    
    print("=" * 80)
    print("프로젝트 구조 분리: data_drive/ + git_code/")
    print("=" * 80)
    
    # 디렉토리 생성
    data_drive_dir.mkdir(exist_ok=True)
    git_code_dir.mkdir(exist_ok=True)
    
    # data_drive로 이동할 항목
    data_drive_items = [
        "data",
        "data_backup",
        "artifacts",
        "LOCAL_TRASH",  # 있으면
    ]
    
    # git_code로 이동할 항목 (나머지 전부)
    git_code_items = [
        ".cursor",
        ".github",
        "configs",
        "docs",
        "notebooks",
        "reports",
        "scripts",
        "src",
        "tests",
        "unused",
        ".gitignore",
        ".pre-commit-config.yaml",
        "LICENSE",
        "Makefile",
        "pyproject.toml",
        "pytest.ini",
        "README.md",
        "README_PORTFOLIO.md",
        "REPRODUCIBILITY_TEST_REPORT.md",
    ]
    
    print("\n[1/3] data_drive/ 생성 중...")
    for item in data_drive_items:
        src = project_root / item
        if src.exists():
            dst = data_drive_dir / item
            if dst.exists():
                print(f"  ⚠️ {item} 이미 존재, 스킵")
            else:
                print(f"  → {item} 이동 중...")
                shutil.move(str(src), str(dst))
                print(f"  ✅ {item} 이동 완료")
        else:
            print(f"  ⚠️ {item} 없음, 스킵")
    
    print("\n[2/3] git_code/ 생성 중...")
    for item in git_code_items:
        src = project_root / item
        if src.exists():
            dst = git_code_dir / item
            if dst.exists():
                print(f"  ⚠️ {item} 이미 존재, 스킵")
            else:
                print(f"  → {item} 이동 중...")
                shutil.move(str(src), str(dst))
                print(f"  ✅ {item} 이동 완료")
        else:
            print(f"  ⚠️ {item} 없음, 스킵")
    
    # data_drive/README.md 생성
    print("\n[3/3] data_drive/README.md 생성 중...")
    data_drive_readme = data_drive_dir / "README.md"
    data_drive_readme.write_text("""# Data Drive

이 폴더는 **데이터 파일 전체**를 포함합니다.

## 📦 포함 내용

- `data/`: 원시 데이터 및 계산 데이터
- `artifacts/`: 실행 아티팩트 (runs/<run_id>/)
- `data_backup/`: 백업 데이터
- `LOCAL_TRASH/`: 정리된 파일들 (있으면)

## 📤 구글 드라이브 업로드

이 폴더 전체를 구글 드라이브에 업로드하세요.

## 🔗 git_code와 함께 사용

`git_code/` 폴더와 함께 사용하여 전체 파이프라인을 재현할 수 있습니다.

```bash
# 구조
test-main/
├── data_drive/    # 이 폴더 (구글 드라이브에서 다운로드)
└── git_code/      # GitHub에서 클론

# 실행
cd git_code
python -m src.pipeline.track_a_pipeline
```

## 📝 참고

- 상세 가이드: `git_code/docs/DATA_EXTERNAL_DRIVE_GUIDE.md`
""", encoding="utf-8")
    print("  ✅ data_drive/README.md 생성 완료")
    
    # git_code/.gitignore에 data_drive 추가
    git_code_gitignore = git_code_dir / ".gitignore"
    if git_code_gitignore.exists():
        content = git_code_gitignore.read_text(encoding="utf-8")
        if "data_drive" not in content:
            content += "\n# Data drive (외부 드라이브에 별도 업로드)\ndata_drive/\n"
            git_code_gitignore.write_text(content, encoding="utf-8")
            print("  ✅ git_code/.gitignore 업데이트 완료")
    
    print("\n" + "=" * 80)
    print("✅ 구조 분리 완료!")
    print("=" * 80)
    print("\n다음 단계:")
    print("  1. git_code/configs/config.yaml 경로 설정 확인")
    print("  2. git_code/README.md 업데이트")
    print("  3. 테스트 실행")


if __name__ == "__main__":
    create_structure()
