"""
프로젝트를 표준 구조로 재구성하는 스크립트

표준 구조:
- src/ (제품 코드)
  - data/ (수집/정제/검증)
  - features/ (피처 생성)
  - models/ (학습/추론)
  - signals/ (스코어→시그널 변환)
  - portfolio/ (포트 구성)
  - backtest/ (백테스트)
  - evaluation/ (리포트/플롯)
  - cli.py (원클릭 실행)
- app/ (UI)
- data/ (raw_data, cal_data)
- artifacts/ (runs/)
- reports/
- docs/ (표준 문서)
- notebooks/
- tests/
- configs/
"""

import sys
from pathlib import Path
import shutil
from collections import defaultdict

project_root = Path(__file__).resolve().parent.parent

print("=" * 80)
print("프로젝트 표준 구조 재구성")
print("=" * 80)

# 1. src/ 구조 분석 및 매핑
print("\n[1] src/ 구조 분석")

src_dir = project_root / "src"
current_modules = {
    "data_collection": "data",
    "tracks/shared/stages/data": "data",
    "tracks/track_a/stages/ranking": "signals",
    "tracks/track_b/stages/backtest": "backtest",
    "tracks/track_b/stages/modeling": "signals",
    "components/ranking": "signals",
    "components/backtest": "backtest",
    "components/portfolio": "portfolio",
    "features": "features",
    "stages/data": "data",
    "stages/ranking": "signals",
    "stages/backtest": "backtest",
    "stages/modeling": "models",
}

# 2. 쓸모없는 파일 분류
print("\n[2] 쓸모없는 파일 분류")

# scripts/에서 임시/테스트 파일 찾기
scripts_dir = project_root / "scripts"
unused_patterns = [
    "test_", "temp_", "check_", "verify_", "find_", "show_",
    "quick_", "simple_", "debug_", "old_", "backup_",
    "recompute_", "cleanup_", "reorganize_",
]

unused_scripts = []
for script_file in scripts_dir.glob("*.py"):
    if any(pattern in script_file.name.lower() for pattern in unused_patterns):
        unused_scripts.append(script_file)

# docs/에서 임시/분석 파일 찾기
docs_dir = project_root / "docs"
unused_docs = []
unused_doc_patterns = ["_analysis_", "_cleanup_", "_inventory_", "_local_"]
for doc_file in docs_dir.glob("*.md"):
    if any(pattern in doc_file.name.lower() for pattern in unused_doc_patterns):
        unused_docs.append(doc_file)

# 루트의 리포트 파일들
root_reports = list(project_root.glob("*_REPORT.md")) + \
               list(project_root.glob("*_AUDIT*.md")) + \
               list(project_root.glob("DATA_*.md"))

print(f"  쓸모없는 scripts: {len(unused_scripts)}개")
print(f"  쓸모없는 docs: {len(unused_docs)}개")
print(f"  루트 리포트: {len(root_reports)}개")

# 3. unused/ 폴더 생성 및 이동
print("\n[3] 쓸모없는 파일 이동")

unused_dir = project_root / "unused"
unused_dir.mkdir(exist_ok=True)
unused_scripts_dir = unused_dir / "scripts"
unused_docs_dir = unused_dir / "docs"
unused_reports_dir = unused_dir / "reports"

unused_scripts_dir.mkdir(parents=True, exist_ok=True)
unused_docs_dir.mkdir(parents=True, exist_ok=True)
unused_reports_dir.mkdir(parents=True, exist_ok=True)

moved_count = 0
for script in unused_scripts:
    try:
        shutil.move(str(script), str(unused_scripts_dir / script.name))
        moved_count += 1
    except Exception as e:
        print(f"  ⚠ 이동 실패: {script.name} - {e}")

for doc in unused_docs:
    try:
        shutil.move(str(doc), str(unused_docs_dir / doc.name))
        moved_count += 1
    except Exception as e:
        print(f"  ⚠ 이동 실패: {doc.name} - {e}")

for report in root_reports:
    try:
        shutil.move(str(report), str(unused_reports_dir / report.name))
        moved_count += 1
    except Exception as e:
        print(f"  ⚠ 이동 실패: {report.name} - {e}")

print(f"  ✓ 이동 완료: {moved_count}개 파일")

# 4. 표준 폴더 구조 생성
print("\n[4] 표준 폴더 구조 생성")

# artifacts/runs/ 생성
artifacts_dir = project_root / "artifacts"
artifacts_runs_dir = artifacts_dir / "runs"
artifacts_runs_dir.mkdir(parents=True, exist_ok=True)

# reports/ 생성
reports_dir = project_root / "reports"
reports_figures_dir = reports_dir / "figures"
reports_figures_dir.mkdir(parents=True, exist_ok=True)

# notebooks/ 생성
notebooks_dir = project_root / "notebooks"
notebooks_dir.mkdir(exist_ok=True)

print("  ✓ artifacts/runs/ 생성")
print("  ✓ reports/figures/ 생성")
print("  ✓ notebooks/ 생성")

# 5. data/ 구조 확인 (이미 raw_data, cal_data로 재구성됨)
print("\n[5] data/ 구조 확인")
data_dir = project_root / "data"
if (data_dir / "raw_data").exists() and (data_dir / "cal_data").exists():
    print("  ✓ data/ 구조 정상 (raw_data, cal_data)")
else:
    print("  ⚠ data/ 구조 확인 필요")

print("\n" + "=" * 80)
print("재구성 완료")
print("=" * 80)
print(f"이동된 파일: {moved_count}개")
print(f"unused/ 폴더: {unused_dir}")
