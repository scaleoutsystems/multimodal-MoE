from pathlib import Path
from src.paths import ZOD_ROOT, ZOD_DINO_DATA


def describe_dir(p: Path, n: int = 20) -> None:
    print(f"\n=== {p} ===")
    print("exists:", p.exists())
    if not p.exists():
        return
    items = sorted(p.iterdir())
    print("num entries:", len(items))
    print("first entries:")
    for x in items[:n]:
        kind = "dir" if x.is_dir() else "file"
        print(f"  - [{kind}] {x.name}")


def main() -> None:
    describe_dir(ZOD_ROOT)
    describe_dir(ZOD_DINO_DATA)


if __name__ == "__main__":
    main()
