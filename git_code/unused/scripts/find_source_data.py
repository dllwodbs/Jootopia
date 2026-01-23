"""원천 데이터 위치 찾기"""
from pathlib import Path
import os

backup = Path("data_backup")
print(f"Searching in: {backup.resolve()}\n")

# 재귀적으로 파일 찾기
source_files = {
    "universe_k200_membership_monthly": [],
    "ohlcv_daily": [],
    "fundamentals_annual": [],
    "panel_merged_daily": [],
}

for root, dirs, files in os.walk(backup):
    root_path = Path(root)
    for file in files:
        name_without_ext = Path(file).stem
        for key in source_files:
            if key in name_without_ext:
                full_path = root_path / file
                source_files[key].append(str(full_path.relative_to(backup)))

print("Found source data files:")
for key, paths in source_files.items():
    if paths:
        print(f"\n{key}:")
        for p in paths[:5]:
            print(f"  - {p}")
    else:
        print(f"\n{key}: NOT FOUND")

# data_backup의 전체 구조 확인
print("\n\n=== data_backup 전체 구조 (최상위 3단계) ===")
def print_tree(path, prefix="", max_depth=3, current_depth=0):
    if current_depth >= max_depth:
        return
    try:
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        for i, item in enumerate(items[:20]):  # 최대 20개만
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            if item.is_dir() and current_depth < max_depth - 1:
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(item, next_prefix, max_depth, current_depth + 1)
    except PermissionError:
        pass

print_tree(backup)
