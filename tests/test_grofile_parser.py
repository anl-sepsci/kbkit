import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from kbkit.parsers import GroFileParser  # Adjust import path as needed

# Sample minimal .gro content
SAMPLE_GRO_CONTENT = """Test GRO file
   3
    1WAT    O     1   0.000   0.000   0.000
    1WAT    H     2   0.100   0.000   0.000
    1WAT    H1    3   0.000   0.100   0.000
    2WAT    O     4   1.000   1.000   1.000
    2WAT    H     5   1.100   1.000   1.000
    2WAT    H1    6   1.000   1.100   1.000
   1.00000  1.00000  1.00000
"""

def create_temp_gro_file(content=SAMPLE_GRO_CONTENT, name="test.gro"):
    temp_dir = TemporaryDirectory()
    gro_path = Path(temp_dir.name) / name
    gro_path.write_text(content)
    return gro_path, temp_dir

def test_valid_gro_file_parsing():
    gro_path, temp_dir = create_temp_gro_file()
    parser = GroFileParser(str(gro_path))

    assert isinstance(parser.electron_count, dict)
    assert "WAT" in parser.electron_count
    assert parser.electron_count["WAT"] > 0

    temp_dir.cleanup()

def test_box_volume_calculation():
    gro_path, temp_dir = create_temp_gro_file()
    parser = GroFileParser(str(gro_path))

    volume = parser.calculate_box_volume()
    assert pytest.approx(volume, 0.001) == 1.0

    temp_dir.cleanup()

def test_invalid_suffix():
    gro_path, temp_dir = create_temp_gro_file(name="test.txt")
    with pytest.raises(ValueError):
        GroFileParser(gro_path)

def test_missing_file():
    with pytest.raises(FileNotFoundError):
        GroFileParser("nonexistent.gro")

def test_invalid_box_line():
    bad_content = SAMPLE_GRO_CONTENT.replace("1.00000  1.00000  1.00000", "invalid box")
    gro_path, temp_dir = create_temp_gro_file(content=bad_content)

    parser = GroFileParser(str(gro_path))
    with pytest.raises(ValueError):
        parser.calculate_box_volume()

    temp_dir.cleanup()
