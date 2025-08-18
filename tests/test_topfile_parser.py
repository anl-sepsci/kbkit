import pytest 
from pathlib import Path 
from kbkit.properties import TopFileParser

@pytest.fixture
def sample_top_file(tmp_path):
    content = """
    ; Sample topology file
    [ molecules ]
    Water     100
    Ethanol   50
    InvalidLine
    Methane   not_a_number
    Acetone   25
    [ atoms ]
    """
    file_path = tmp_path / "sample.top"
    file_path.write_text(content.strip())
    return file_path

def test_parse_valid_molecules(sample_top_file):
    parser = TopFileParser(sample_top_file, verbose=False)
    result = parser.parse()
    assert result == {
        "Water": 100,
        "Ethanol": 50,
        "Acetone": 25
    }

def test_skipped_lines(sample_top_file):
    parser = TopFileParser(sample_top_file)
    parser.parse()
    assert len(parser.skipped_lines) == 2
    skipped_reasons = [reason for _, reason in parser.skipped_lines]
    assert "Missing molecule name or count" in skipped_reasons
    assert "Invalid molecule count" in skipped_reasons

def test_empty_file_raises(tmp_path):
    empty_file = tmp_path / "empty.top"
    empty_file.write_text("")
    parser = TopFileParser(empty_file)
    with pytest.raises(ValueError, match="No molecules found"):
        parser.parse()

