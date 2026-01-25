"""
Complete test coverage for kbkit.utils.validation module.
Target: >95% coverage
"""
import warnings

# Suppress NumPy/SciPy compatibility warning (harmless with NumPy 2.x + SciPy 1.16+)
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)

import os
from pathlib import Path

import pytest

from kbkit.utils.validation import GRO_MIN_LENGTH, validate_path


class TestValidatePath:
    """Test validate_path function."""

    def test_valid_directory_path(self, test_data_dir):
        """Test validation of a valid directory path."""
        result = validate_path(test_data_dir)
        assert isinstance(result, Path)
        assert result.is_dir()
        assert result.exists()

    def test_valid_file_path_no_suffix(self, temp_file):
        """Test validation of a file path without suffix check."""
        # When no suffix is specified, it expects a directory
        # So we need to test with a directory
        result = validate_path(temp_file.parent)
        assert isinstance(result, Path)
        assert result.is_dir()

    def test_valid_file_with_suffix(self, test_data_dir):
        """Test validation of a file with correct suffix."""
        # Create a test file with .txt suffix
        test_file = test_data_dir / "test.txt"
        test_file.write_text("test content")

        result = validate_path(test_file, suffix=".txt")
        assert isinstance(result, Path)
        assert result.is_file()
        assert result.suffix == ".txt"

    def test_valid_gro_file(self, test_data_dir):
        """Test validation of a valid .gro file."""
        gro_file = test_data_dir / "test.gro"
        # Create a valid .gro file with minimum required lines
        content = """Title




    3
        1SOL    OW    1   0.126   0.126   0.126
        1SOL   HW1    2   0.190   0.126   0.126
        1SOL   HW2    3   0.126   0.190   0.126
    1.86206   1.86206   1.86206
    """
        gro_file.write_text(content)



        result = validate_path(gro_file, suffix=".gro")
        assert isinstance(result, Path)
        assert result.suffix == ".gro"

    def test_string_path_input(self, test_data_dir):
        """Test that string paths are accepted."""
        result = validate_path(str(test_data_dir))
        assert isinstance(result, Path)
        assert result.is_dir()

    def test_path_object_input(self, test_data_dir):
        """Test that Path objects are accepted."""
        result = validate_path(test_data_dir)
        assert isinstance(result, Path)
        assert result.is_dir()

    def test_path_resolution(self, test_data_dir):
        """Test that paths are resolved (symlinks, relative paths)."""
        # Create a file in test directory
        test_file = test_data_dir / "test.txt"
        test_file.write_text("content")

        # Use relative path
        original_dir = os.getcwd()
        try:
            os.chdir(test_data_dir.parent)
            relative_path = test_data_dir.name
            result = validate_path(relative_path)

            # Should be resolved to absolute path
            assert result.is_absolute()
        finally:
            os.chdir(original_dir)

    def test_invalid_type_raises_error(self):
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError, match="Expected a string path"):
            validate_path(123)

        with pytest.raises(TypeError, match="Expected a string path"):
            validate_path(["/path/to/dir"])

        with pytest.raises(TypeError, match="Expected a string path"):
            validate_path(None)

    def test_nonexistent_file_with_suffix(self, test_data_dir):
        """Test that nonexistent file raises FileNotFoundError."""
        nonexistent = test_data_dir / "nonexistent.txt"

        with pytest.raises(FileNotFoundError, match="Path is not a file"):
            validate_path(nonexistent, suffix=".txt")

    def test_wrong_suffix_raises_error(self, test_data_dir):
        """Test that wrong suffix raises ValueError."""
        test_file = test_data_dir / "test.txt"
        test_file.write_text("content")

        with pytest.raises(ValueError, match="Suffix .gro does not match file suffix"):
            validate_path(test_file, suffix=".gro")

    def test_directory_with_suffix_raises_error(self, test_data_dir):
        """Test that directory path with suffix requirement raises error."""
        with pytest.raises(FileNotFoundError, match="Path is not a file"):
            validate_path(test_data_dir, suffix=".txt")

    def test_file_without_suffix_raises_error(self, test_data_dir):
        """Test that file path without suffix (expecting directory) raises error."""
        test_file = test_data_dir / "test.txt"
        test_file.write_text("content")

        with pytest.raises(ValueError, match="Path is not a directory"):
            validate_path(test_file)

    def test_gro_file_too_short(self, test_data_dir):
        """Test that .gro file with too few lines raises ValueError."""
        gro_file = test_data_dir / "short.gro"
        # Create a .gro file with fewer than GRO_MIN_LENGTH lines
        gro_file.write_text("Line 1\nLine 2\n")

        with pytest.raises(ValueError, match="too short to be a valid .gro file"):
            validate_path(gro_file, suffix=".gro")

    def test_gro_file_exactly_min_length(self, test_data_dir):
        """Test .gro file with exactly minimum required lines."""
        gro_file = test_data_dir / "min.gro"
        # Create exactly GRO_MIN_LENGTH lines
        lines = "\n".join([f"Line {i}" for i in range(GRO_MIN_LENGTH)])
        gro_file.write_text(lines)

        result = validate_path(gro_file, suffix=".gro")
        assert isinstance(result, Path)

    def test_nonexistent_directory_raises_error(self):
        """Test that nonexistent directory raises ValueError."""
        with pytest.raises(ValueError, match="Path is not a directory"):
            validate_path("/nonexistent/directory/path")

    def test_permission_error(self, test_data_dir, monkeypatch):
        """Test that unreadable path raises PermissionError."""
        # Mock os.access to return False
        def mock_access(path, mode):
            return False

        monkeypatch.setattr(os, "access", mock_access)

        with pytest.raises(PermissionError, match="Cannot read files in path"):
            validate_path(test_data_dir)

    def test_various_file_suffixes(self, test_data_dir):
        """Test validation with various file suffixes."""
        suffixes = [".txt", ".csv", ".json", ".dat", ".log"]

        for suffix in suffixes:
            test_file = test_data_dir / f"test{suffix}"
            test_file.write_text("content")

            result = validate_path(test_file, suffix=suffix)
            assert result.suffix == suffix

    def test_empty_suffix_string(self, test_data_dir):
        """Test with empty suffix string (should treat as no suffix)."""
        # Empty suffix means expecting directory
        result = validate_path(test_data_dir, suffix="")
        assert result.is_dir()

    def test_path_with_spaces(self, test_data_dir):
        """Test path with spaces in name."""
        dir_with_spaces = test_data_dir / "dir with spaces"
        dir_with_spaces.mkdir()

        result = validate_path(dir_with_spaces)
        assert result.is_dir()
        assert "spaces" in str(result)

    def test_path_with_special_characters(self, test_data_dir):
        """Test path with special characters."""
        special_dir = test_data_dir / "dir-with_special.chars"
        special_dir.mkdir()

        result = validate_path(special_dir)
        assert result.is_dir()




class TestGroMinLength:
    """Test GRO_MIN_LENGTH constant."""

    def test_gro_min_length_value(self):
        """Test that GRO_MIN_LENGTH has expected value."""
        assert GRO_MIN_LENGTH == 3

    def test_gro_min_length_is_int(self):
        """Test that GRO_MIN_LENGTH is an integer."""
        assert isinstance(GRO_MIN_LENGTH, int)

    def test_gro_min_length_positive(self):
        """Test that GRO_MIN_LENGTH is positive."""
        assert GRO_MIN_LENGTH > 0




class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_hidden_directory(self, test_data_dir):
        """Test validation of hidden directory (starts with .)."""
        hidden_dir = test_data_dir / ".hidden"
        hidden_dir.mkdir()

        result = validate_path(hidden_dir)
        assert result.is_dir()

    def test_hidden_file(self, test_data_dir):
        """Test validation of hidden file."""
        hidden_file = test_data_dir / ".hidden.txt"
        hidden_file.write_text("content")

        result = validate_path(hidden_file, suffix=".txt")
        assert result.is_file()

    def test_nested_directory_structure(self, test_data_dir):
        """Test deeply nested directory structure."""
        nested = test_data_dir / "a" / "b" / "c" / "d"
        nested.mkdir(parents=True)

        result = validate_path(nested)
        assert result.is_dir()

    def test_file_with_multiple_dots(self, test_data_dir):
        """Test file with multiple dots in name."""
        multi_dot_file = test_data_dir / "file.name.with.dots.txt"
        multi_dot_file.write_text("content")

        result = validate_path(multi_dot_file, suffix=".txt")
        assert result.suffix == ".txt"

    def test_file_without_extension(self, test_data_dir):
        """Test file without extension."""
        no_ext_file = test_data_dir / "README"
        no_ext_file.write_text("content")

        # Should work if we don't specify suffix
        # But since no suffix means expecting directory, this will fail
        with pytest.raises(ValueError, match="Path is not a directory"):
            validate_path(no_ext_file)

    def test_very_long_path(self, test_data_dir):
        """Test with very long path name."""
        long_name = "a" * 100
        long_dir = test_data_dir / long_name
        long_dir.mkdir()

        result = validate_path(long_dir)
        assert result.is_dir()

    def test_unicode_in_path(self, test_data_dir):
        """Test path with unicode characters."""
        unicode_dir = test_data_dir / "dir_with_unicode_测试"
        unicode_dir.mkdir()

        result = validate_path(unicode_dir)
        assert result.is_dir()

    def test_gro_file_with_empty_lines(self, test_data_dir):
        """Test .gro file with empty lines."""
        gro_file = test_data_dir / "empty_lines.gro"
        content = """Title

    1SOL    OW    1   0.126   0.126   0.126




    1.86206   1.86206   1.86206
    """
        gro_file.write_text(content)



        # Should still count all lines including empty ones
        result = validate_path(gro_file, suffix=".gro")
        assert isinstance(result, Path)

    def test_case_sensitive_suffix(self, test_data_dir):
        """Test that suffix matching is case-sensitive."""
        test_file = test_data_dir / "test.TXT"
        test_file.write_text("content")

        # Should fail because .TXT != .txt
        with pytest.raises(ValueError, match="Suffix .txt does not match"):
            validate_path(test_file, suffix=".txt")

        # Should succeed with correct case
        result = validate_path(test_file, suffix=".TXT")
        assert result.suffix == ".TXT"




    class TestIntegration:
        """Integration tests for validate_path."""



    def test_validate_multiple_files_in_directory(self, test_data_dir):
        """Test validating multiple files in a directory."""
        # Create multiple files
        files = []
        for i in range(5):
            f = test_data_dir / f"file{i}.txt"
            f.write_text(f"content {i}")
            files.append(f)

        # Validate each file
        for f in files:
            result = validate_path(f, suffix=".txt")
            assert result.is_file()

    def test_validate_directory_then_file(self, test_data_dir):
        """Test validating directory then file within it."""
        # First validate directory
        dir_result = validate_path(test_data_dir)
        assert dir_result.is_dir()

        # Create and validate file within
        test_file = test_data_dir / "test.txt"
        test_file.write_text("content")
        file_result = validate_path(test_file, suffix=".txt")
        assert file_result.is_file()
        assert file_result.parent == dir_result

    def test_workflow_with_gro_files(self, test_data_dir):
        """Test typical workflow with .gro files."""
        # Create valid .gro file
        gro_file = test_data_dir / "system.gro"
        content = """System




    10
    """ + "\n".join([f"    1SOL    OW    {i}   0.1   0.1   0.1" for i in range(1, 11)]) + "\n   1.0   1.0   1.0\n"
        gro_file.write_text(content)



        # Validate
        result = validate_path(gro_file, suffix=".gro")
        assert result.is_file()
        assert result.suffix == ".gro"
        assert len(result.read_text().splitlines()) >= GRO_MIN_LENGTH

    def test_error_messages_are_informative(self, test_data_dir):
        """Test that error messages contain useful information."""
        test_file = test_data_dir / "test.txt"
        test_file.write_text("content")

        # Wrong suffix
        try:
            validate_path(test_file, suffix=".gro")
        except ValueError as e:
            assert ".gro" in str(e)
            assert ".txt" in str(e)

        # File instead of directory
        try:
            validate_path(test_file)
        except ValueError as e:
            assert "not a directory" in str(e)
            assert str(test_file) in str(e)


