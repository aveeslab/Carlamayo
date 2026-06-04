from pathlib import Path


def test_docs_use_direct_carla_unreal_launch():
    docs = [Path("README.md"), *sorted(Path("docs").glob("*.md"))]
    for path in docs:
        text = path.read_text()
        assert "scripts/start_carla_010.sh" not in text
        assert "./scripts/start" not in text
