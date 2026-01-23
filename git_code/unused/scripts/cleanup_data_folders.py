"""
data 폴더 정리 스크립트

최종 구조(raw_data, cal_data, README.md)를 제외한 다른 폴더/파일 삭제
"""

import sys
from pathlib import Path
import shutil

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_path

cfg = load_config("configs/config.yaml")
data_dir = Path(get_path(cfg, "data_interim")).parent

# 유지할 항목
KEEP_ITEMS = {"raw_data", "cal_data", "README.md"}

print("=" * 80)
print("data 폴더 정리")
print("=" * 80)
print(f"대상 폴더: {data_dir}")
print(f"유지할 항목: {KEEP_ITEMS}")
print()

# 삭제할 항목 확인
items_to_delete = []
for item in data_dir.iterdir():
    if item.name not in KEEP_ITEMS:
        items_to_delete.append(item)

if not items_to_delete:
    print("삭제할 항목이 없습니다.")
    sys.exit(0)

print(f"[삭제할 항목] ({len(items_to_delete)}개)")
for item in items_to_delete:
    item_type = "폴더" if item.is_dir() else "파일"
    print(f"  - {item.name} ({item_type})")

print()
response = input("삭제하시겠습니까? (yes/no): ")

if response.lower() != "yes":
    print("취소되었습니다.")
    sys.exit(0)

# 삭제 실행
print()
print("[삭제 실행]")
deleted_count = 0
for item in items_to_delete:
    try:
        if item.is_dir():
            shutil.rmtree(item)
            print(f"  ✓ 폴더 삭제: {item.name}")
        else:
            item.unlink()
            print(f"  ✓ 파일 삭제: {item.name}")
        deleted_count += 1
    except Exception as e:
        print(f"  ✗ 삭제 실패: {item.name} - {e}")

print()
print("=" * 80)
print(f"삭제 완료: {deleted_count}개 항목")
print("=" * 80)

# 최종 구조 확인
print()
print("[최종 구조]")
for item in sorted(data_dir.iterdir()):
    item_type = "📁" if item.is_dir() else "📄"
    print(f"  {item_type} {item.name}")
