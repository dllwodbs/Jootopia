"""data_backup 폴더 구조 확인"""
from pathlib import Path

backup = Path("data_backup")
print(f"data_backup exists: {backup.exists()}")

if backup.exists():
    print("\nDirectories:")
    dirs = [d.name for d in backup.iterdir() if d.is_dir()]
    for d in dirs[:10]:
        print(f"  - {d}")
    
    print("\nChecking data_backup/data/interim:")
    interim = backup / "data" / "interim"
    if interim.exists():
        files = [f.name for f in interim.iterdir() if f.is_file()]
        print(f"  Found {len(files)} files:")
        for f in files[:20]:
            print(f"    - {f}")
    else:
        print("  data_backup/data/interim does not exist")
        
    # 다른 경로 확인
    print("\nChecking alternative paths:")
    for alt_path in [backup / "interim", backup / "data_backup" / "interim"]:
        if alt_path.exists():
            files = [f.name for f in alt_path.iterdir() if f.is_file()]
            print(f"  {alt_path}: {len(files)} files")
