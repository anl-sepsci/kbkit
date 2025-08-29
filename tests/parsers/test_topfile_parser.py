"""
Tests for TopFileParser's molecule section.

Covers:
- Parsing valid molecule entries into name-count mappings
- Skipping malformed lines with informative reasons
- Raising errors on empty files with no molecule data

Uses a sample topology file with mixed valid and invalid entries.
"""

import pytest

from kbkit.parsers.top_file import TopFileParser


@pytest.fixture
def sample_top_file(tmp_path):
    """
    Create a temporary topology file with a mix of valid and invalid molecule entries.

    Returns
    -------
    Path
        Path to the temporary .top file.
    """
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


EXPECTED_SKIPPED_LINES = 2  # number of invalid lines in sample_top_file


def test_parse_valid_molecules(sample_top_file):
    """
    Test that valid molecule entries are correctly parsed from the topology file.

    Asserts that only well-formed lines with valid molecule names and integer counts
    are included in the result.
    """
    parser = TopFileParser(sample_top_file, verbose=False)
    result = parser.molecule_count
    assert result == {"Water": 100, "Ethanol": 50, "Acetone": 25}


def test_skipped_lines(sample_top_file):
    """
    Test that malformed lines are skipped and logged with appropriate reasons.

    Verifies that two lines are skipped:
    - One with missing count
    - One with a non-numeric count
    """
    parser = TopFileParser(sample_top_file)
    parser.parse()
    assert len(parser.skipped_lines) == EXPECTED_SKIPPED_LINES
    skipped_reasons = [reason for _, reason in parser.skipped_lines]
    assert "Missing molecule name or count" in skipped_reasons
    assert "Invalid molecule count" in skipped_reasons


def test_empty_file_raises(tmp_path):
    """
    Test that parsing an empty topology file raises a ValueError.

    Ensures that the parser fails gracefully when no molecule data is present.
    """
    empty_file = tmp_path / "empty.top"
    empty_file.write_text("")
    parser = TopFileParser(empty_file)
    with pytest.raises(ValueError, match="No molecules found"):
        parser.parse()
