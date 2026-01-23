"""
분리된 구조(data_drive + git_code)에 맞게 경로 설정 업데이트

구조:
test-main/
├── data_drive/    # 데이터 (구글 드라이브)
└── git_code/      # 코드 (GitHub)
"""

import sys
from pathlib import Path
import yaml

# git_code/scripts/에서 실행 중
git_code_dir = Path(__file__).resolve().parent.parent
project_root = git_code_dir.parent
data_drive_dir = project_root / "data_drive"


def update_config_yaml():
    """config.yaml 경로 설정 업데이트"""
    
    config_path = git_code_dir / "configs" / "config.yaml"
    
    if not config_path.exists():
        print(f"⚠️ {config_path} 없음, 스킵")
        return
    
    print(f"[경로 설정 업데이트] {config_path}")
    
    # YAML 로드
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    
    # paths 섹션 업데이트
    if "paths" not in cfg:
        cfg["paths"] = {}
    
    # base_dir을 git_code로 설정
    cfg["paths"]["base_dir"] = str(git_code_dir.resolve())
    
    # data 경로를 ../data_drive로 설정
    cfg["paths"]["data_cal"] = str((data_drive_dir / "data" / "cal_data").resolve())
    cfg["paths"]["data_raw_data"] = str((data_drive_dir / "data" / "raw_data").resolve())
    cfg["paths"]["data_interim"] = str((data_drive_dir / "data" / "cal_data").resolve())  # cal_data를 interim으로도 사용
    
    # artifacts 경로
    cfg["paths"]["artifacts_models"] = str((data_drive_dir / "artifacts" / "models").resolve())
    cfg["paths"]["artifacts_reports"] = str((data_drive_dir / "artifacts" / "reports").resolve())
    
    # logs는 git_code 내부에
    cfg["paths"]["logs"] = str((git_code_dir / "logs").resolve())
    
    # 저장
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    print("  ✅ config.yaml 업데이트 완료")
    print(f"     base_dir: {cfg['paths']['base_dir']}")
    print(f"     data_cal: {cfg['paths']['data_cal']}")
    print(f"     data_raw_data: {cfg['paths']['data_raw_data']}")


def create_setup_script():
    """설정 스크립트 생성"""
    
    setup_script = git_code_dir / "setup_paths.py"
    
    content = '''"""
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
'''
    
    setup_script.write_text(content, encoding="utf-8")
    print(f"  ✅ {setup_script} 생성 완료")


def main():
    """메인 함수"""
    print("=" * 80)
    print("경로 설정 업데이트")
    print("=" * 80)
    
    if not git_code_dir.exists():
        print("❌ git_code/ 폴더가 없습니다. 먼저 구조 분리를 실행하세요.")
        print("   python scripts/restructure_to_data_drive_git_code.py")
        return
    
    update_config_yaml()
    create_setup_script()
    
    print("\n" + "=" * 80)
    print("✅ 경로 설정 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
