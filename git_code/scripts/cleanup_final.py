"""
최종 정리 스크립트 - 불필요한 파일들을 archive 폴더로 이동
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """프로젝트 정리"""
    archive_dir = Path("archive")
    archive_dir.mkdir(exist_ok=True)

    # 정리할 파일들
    cleanup_items = [
        # 기존 데이터 폴더들
        "interim",
        "external",
        "processed",

        # 임시 파일들
        "LOCAL_TRASH",
        "backup_repro_test",
        "raw",

        # 개별 파일들
        "kospi200_benchmark_cumulative_returns.csv",
        "sample_data_readme.md",
    ]

    print("프로젝트 정리 시작...")

    moved_count = 0
    for item in cleanup_items:
        item_path = Path(item)
        if item_path.exists():
            try:
                dest_path = archive_dir / item_path.name
                if dest_path.exists():
                    # 이름 충돌 시 번호 추가
                    counter = 1
                    stem = dest_path.stem
                    suffix = dest_path.suffix or ""
                    while dest_path.exists():
                        dest_path = archive_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                if item_path.is_file():
                    shutil.move(str(item_path), str(dest_path))
                    print(f"  ✓ 파일 이동: {item_path} → {dest_path}")
                else:
                    shutil.move(str(item_path), str(dest_path))
                    print(f"  ✓ 폴더 이동: {item_path}/ → {dest_path}/")
                moved_count += 1

            except Exception as e:
                print(f"  ✗ 이동 실패: {item_path} - {e}")

    # scripts 폴더 정리
    scripts_dir = Path("scripts")
    keep_scripts = {
        "recompute_calculated_data.py",
        "recompute_step_by_step.py",
        "show_track_b_metrics.py",
        "show_ranking.py",
        "recompute_track_a_with_integrated.py",
        "check_recompute_results.py",
        "organize_project.py",
        "cleanup_final.py",
    }

    if scripts_dir.exists():
        old_scripts_dir = archive_dir / "old_scripts"
        old_scripts_dir.mkdir(exist_ok=True)

        for script_file in scripts_dir.glob("*"):
            if script_file.is_file() and script_file.name not in keep_scripts:
                try:
                    dest_path = old_scripts_dir / script_file.name
                    shutil.move(str(script_file), str(dest_path))
                    print(f"  ✓ 스크립트 이동: {script_file} → {dest_path}")
                    moved_count += 1
                except Exception as e:
                    print(f"  ✗ 스크립트 이동 실패: {script_file} - {e}")

    print(f"\n정리 완료! 총 {moved_count}개 항목 이동")
    print(f"아카이브 폴더: {archive_dir}")

if __name__ == "__main__":
    cleanup_project()