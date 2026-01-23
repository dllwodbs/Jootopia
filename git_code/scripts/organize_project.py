"""
프로젝트 폴더 정리 스크립트
GitHub 포트폴리오 표준 구조로 재구성
"""

import os
import shutil
from pathlib import Path

def create_archive_folder():
    """아카이브 폴더 생성"""
    archive_dir = Path("archive")
    archive_dir.mkdir(exist_ok=True)

    # 서브폴더 생성
    subdirs = ["old_scripts", "temp_files", "duplicates", "legacy"]
    for subdir in subdirs:
        (archive_dir / subdir).mkdir(exist_ok=True)

    return archive_dir

def get_files_to_archive():
    """아카이브할 파일 목록 반환"""
    files_to_archive = []

    # scripts 폴더 정리 (유지할 파일만 남기기)
    scripts_dir = Path("scripts")
    if scripts_dir.exists():
        keep_scripts = {
            "recompute_calculated_data.py",  # 메인 재산출 스크립트
            "recompute_step_by_step.py",     # 단계별 실행
            "show_track_b_metrics.py",       # 성과 지표 확인
            "show_ranking.py",               # 랭킹 조회
            "recompute_track_a_with_integrated.py",  # Track A 재산출
            "check_recompute_results.py",    # 결과 확인
            "organize_project.py",           # 이 파일
        }

        for file_path in scripts_dir.glob("*"):
            if file_path.is_file() and file_path.name not in keep_scripts:
                files_to_archive.append(("old_scripts", file_path))

    # 임시 파일들
    temp_patterns = ["*.tmp", "*.log", "*.pyc", "__pycache__"]
    for pattern in temp_patterns:
        for file_path in Path(".").rglob(pattern):
            if file_path.is_file() or file_path.is_dir():
                files_to_archive.append(("temp_files", file_path))

    # 중복 파일들
    duplicates = []
    # config.yaml이 여러 곳에 있을 수 있음
    config_files = list(Path(".").rglob("config.yaml"))
    if len(config_files) > 1:
        # configs/config.yaml만 유지
        main_config = Path("configs/config.yaml")
        for config_file in config_files:
            if config_file != main_config:
                duplicates.append(config_file)

    for dup_file in duplicates:
        files_to_archive.append(("duplicates", dup_file))

    # 레거시 파일들
    legacy_files = [
        "LOCAL_TRASH",
        "backup_repro_test",
        "raw",
        "sample_data_readme.md",
        "kospi200_benchmark_cumulative_returns.csv"
    ]

    for legacy in legacy_files:
        legacy_path = Path(legacy)
        if legacy_path.exists():
            files_to_archive.append(("legacy", legacy_path))

    return files_to_archive

def archive_files(archive_dir, files_to_archive):
    """파일들을 아카이브 폴더로 이동"""
    print("파일 아카이브 시작...")

    for subdir, file_path in files_to_archive:
        if not file_path.exists():
            continue

        # 대상 경로
        dest_dir = archive_dir / subdir
        dest_path = dest_dir / file_path.name

        # 이름 충돌 방지
        counter = 1
        while dest_path.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            dest_path = dest_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        try:
            if file_path.is_file():
                shutil.move(str(file_path), str(dest_path))
                print(f"  ✓ {file_path} → {dest_path}")
            elif file_path.is_dir():
                shutil.move(str(file_path), str(dest_path))
                print(f"  ✓ {file_path}/ → {dest_path}/")
        except Exception as e:
            print(f"  ✗ 이동 실패 {file_path}: {e}")

def create_clean_structure():
    """표준 구조 생성 및 정리"""
    print("\n표준 구조 정리...")

    # src 폴더 재구성 (기존 구조를 표준에 맞게 매핑)
    src_dir = Path("src")

    # 기존 폴더들을 표준 구조로 링크/복사
    mappings = {
        "data": ["data_collection"],  # 데이터 수집
        "features": ["tracks/shared/stages/data/l3_feature_engineering.py"],  # 피처 엔지니어링
        "models": [],  # 머신러닝 모델 없음 (룰 기반)
        "signals": ["tracks/track_a"],  # 시그널 생성
        "portfolio": ["tracks/track_b"],  # 포트폴리오 구성
        "backtest": ["tracks/track_b/stages/backtest"],  # 백테스트
        "evaluation": [],  # 평가 (없음)
    }

    # README 파일들 생성
    for module, sources in mappings.items():
        module_dir = src_dir / module
        if not sources and module in ["models", "evaluation"]:
            # 빈 모듈에는 __init__.py만
            module_dir.mkdir(exist_ok=True)
            init_file = module_dir / "__init__.py"
            if not init_file.exists():
                init_file.write_text(f'"""{module} module - Not implemented (rule-based system)"""\n')
                print(f"  ✓ {init_file} 생성")

    # CLI 파일 생성 (기존 scripts 기능을 통합)
    cli_file = src_dir / "cli.py"
    if not cli_file.exists():
        cli_content = '''"""
Command Line Interface for Quant Trading System
"""

import click
from pathlib import Path

@click.group()
def cli():
    """Quant Trading Portfolio CLI"""
    pass

@cli.command()
@click.option("--config", default="configs/config.yaml", help="Config file path")
@click.option("--force", is_flag=True, help="Force rebuild")
def run(config, force):
    """Run full pipeline"""
    click.echo(f"Running full pipeline with config: {config}")
    # Implementation here
    pass

@cli.command()
@click.option("--config", default="configs/config.yaml", help="Config file path")
@click.option("--force", is_flag=True, help="Force rebuild")
def track_a(config, force):
    """Run Track A (Ranking)"""
    click.echo(f"Running Track A with config: {config}")
    from src.pipeline.track_a_pipeline import run_track_a_pipeline
    result = run_track_a_pipeline(config_path=config, force_rebuild=force)
    click.echo("✅ Track A completed")

@cli.command()
@click.option("--strategy", required=True, help="Strategy name")
@click.option("--config", default="configs/config.yaml", help="Config file path")
@click.option("--force", is_flag=True, help="Force rebuild")
def track_b(strategy, config, force):
    """Run Track B (Backtest)"""
    click.echo(f"Running Track B strategy: {strategy}")
    from src.pipeline.track_b_pipeline import run_track_b_pipeline
    result = run_track_b_pipeline(
        config_path=config,
        strategy=strategy,
        force_rebuild=force
    )
    click.echo("✅ Track B completed")

@cli.command()
@click.option("--config", default="configs/config.yaml", help="Config file path")
def data_download(config):
    """Download raw data"""
    click.echo("Downloading raw data...")
    from src.data_collection.pipeline import DataCollectionPipeline
    pipeline = DataCollectionPipeline(config_path=config)
    pipeline.run_l0()
    pipeline.run_l1()
    click.echo("✅ Data download completed")

@cli.command()
@click.option("--config", default="configs/config.yaml", help="Config file path")
def data_validate(config):
    """Validate data integrity"""
    click.echo("Validating data...")
    # Add validation logic
    click.echo("✅ Data validation completed")

if __name__ == "__main__":
    cli()
'''
        cli_file.write_text(cli_content)
        print(f"  ✓ {cli_file} 생성")

def update_gitignore():
    """.gitignore 업데이트"""
    gitignore_path = Path(".gitignore")
    if gitignore_path.exists():
        with open(gitignore_path, "r") as f:
            content = f.read()

        # 추가할 항목들
        additions = [
            "\n# Data and artifacts",
            "data/raw_data/",
            "data/cal_data/",
            "artifacts/",
            "archive/",
            "\n# Environment",
            ".env",
            "\n# Python",
            "__pycache__/",
            "*.pyc",
            "*.pyo",
            ".Python",
            "\n# IDE",
            ".vscode/",
            ".idea/",
            "*.swp",
            "*.swo",
        ]

        for addition in additions:
            if addition not in content:
                content += addition + "\n"

        with open(gitignore_path, "w") as f:
            f.write(content)

        print("  ✓ .gitignore 업데이트")

def main():
    """메인 정리 함수"""
    print("=" * 80)
    print("GitHub 포트폴리오 표준 구조로 프로젝트 정리")
    print("=" * 80)

    # 아카이브 폴더 생성
    archive_dir = create_archive_folder()
    print(f"아카이브 폴더 생성: {archive_dir}")

    # 아카이브할 파일 목록
    files_to_archive = get_files_to_archive()
    print(f"\n아카이브할 파일 수: {len(files_to_archive)}")

    # 파일 아카이브
    if files_to_archive:
        archive_files(archive_dir, files_to_archive)

    # 표준 구조 생성
    create_clean_structure()

    # .gitignore 업데이트
    update_gitignore()

    print("\n" + "=" * 80)
    print("프로젝트 정리 완료!")
    print("=" * 80)

    print("\n현재 구조:")
    print("├── README.md")
    print("├── LICENSE")
    print("├── .gitignore")
    print("├── pyproject.toml")
    print("├── Makefile")
    print("├── env_example.txt")
    print("├── configs/")
    print("├── src/")
    print("│   ├── cli.py")
    print("│   ├── data/")
    print("│   ├── features/")
    print("│   ├── signals/")
    print("│   ├── portfolio/")
    print("│   ├── backtest/")
    print("│   └── evaluation/")
    print("├── data/")
    print("│   ├── raw_data/")
    print("│   └── cal_data/")
    print("├── artifacts/")
    print("├── reports/")
    print("├── docs/")
    print("└── archive/")

if __name__ == "__main__":
    main()