import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from kbkit.properties import GroFileParser  # Adjust import path as needed

# Sample minimal .gro content
SAMPLE_GRO_CONTENT = """Test GRO file
   3
    1WAT     OW    1   0.000   0.000   0.000
    1WAT    HW1    2   0.100   0.000   0.000
    1WAT    HW2    3   0.000   0.100   0.000
   1.00000  1.00000  1.00000
"""

def create_temp_gro_file(content=SAMPLE_GRO_CONTENT):
    temp_dir = TemporaryDirectory()
    gro_path = Path(temp_dir.name) / "test.gro"
    gro_path.write_text(content)
    return gro_path, temp_dir

def test_valid_gro_file_parsing():
    gro_path, temp_dir = create_temp_gro_file()
    parser = GroFileParser(str(gro_path))

    assert isinstance(parser.electron_dict, dict)
    assert "WAT" in parser.electron_dict
    assert parser.electron_dict["WAT"] > 0

    temp_dir.cleanup()

def test_box_volume_calculation():
    gro_path, temp_dir = create_temp_gro_file()
    parser = GroFileParser(str(gro_path))

    volume = parser.calculate_box_volume()
    assert pytest.approx(volume, 0.001) == 1.0

    temp_dir.cleanup()

def test_invalid_file_extension():
    with pytest.raises(ValueError):
        GroFileParser("invalid.txt")

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
