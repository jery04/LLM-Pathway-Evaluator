from pathlib import Path
import shutil

import kagglehub


def copy_dataset_to_data(source_dir: Path, target_dir: Path) -> None:
    """Copy downloaded dataset contents into the project's data folder."""
    target_dir.mkdir(parents=True, exist_ok=True)

    for item in source_dir.iterdir():
        destination = target_dir / item.name

        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    # Download latest version from KaggleHub cache.
    path = kagglehub.dataset_download("everydaycodings/multi-platform-online-courses-dataset")
    source_path = Path(path)

    print(f"Path to dataset files: {source_path}")

    copy_dataset_to_data(source_path, data_dir)
    print(f"Dataset copied to: {data_dir}")


if __name__ == "__main__":
    main()
