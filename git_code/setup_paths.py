"""
경로 자동 설정 스크립트

data_drive와 git_code를 합쳤을 때 자동으로 경로를 설정합니다.
"""

import sys
from pathlib import Path

# git_code에서 실행 중인지 확인
git_code_dir = Path(__file__).resolve().parent
project_root = git_code_dir.parent
data_drive_dir = project_root / "data_drive"

# data_drive 존재 확인
if not data_drive_dir.exists():
    print("⚠️ data_drive/ 폴더가 없습니다.")
    print("   구글 드라이브에서 data_drive 폴더를 다운로드하세요.")
    sys.exit(1)

# 환경 변수 설정 (선택적)
import os
os.environ["DATA_DRIVE_PATH"] = str(data_drive_dir.resolve())
os.environ["BASE_DIR"] = str(git_code_dir.resolve())

print("✅ 경로 설정 완료")
print(f"   BASE_DIR: {git_code_dir.resolve()}")
print(f"   DATA_DRIVE_PATH: {data_drive_dir.resolve()}")
