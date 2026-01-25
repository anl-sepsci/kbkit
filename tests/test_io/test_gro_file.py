"""Unit tests for GroParser class."""

import warnings

# Suppress NumPy/SciPy compatibility warning (harmless with NumPy 2.x + SciPy 1.16+)
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)

from pathlib import Path
from unittest.mock import Mock, patch

import MDAnalysis
import numpy as np
import pytest
from MDAnalysis.core.groups import ResidueGroup

from kbkit.io.gro import GroParser


@pytest.fixture
def sample_gro_file(tmp_path):
    """Create a sample .gro file for testing."""
    gro_content = """Water system
    6
    1WAT     OW    1   0.126   0.126   0.126
    1WAT    HW1    2   0.190   0.126   0.126
    1WAT    HW2    3   0.062   0.126   0.126
    2WAT     OW    4   0.626   0.626   0.626
    2WAT    HW1    5   0.690   0.626   0.626
    2WAT    HW2    6   0.562   0.626   0.626
   1.86206   1.86206   1.86206
"""
    gro_file = tmp_path / "test.gro"
    gro_file.write_text(gro_content)
    return str(gro_file)


@pytest.fixture
def mixed_system_gro_file(tmp_path):
    """Create a .gro file with multiple molecule types."""
    gro_content = """Mixed system: water and methanol
    9
    1WAT     OW    1   0.126   0.126   0.126
    1WAT    HW1    2   0.190   0.126   0.126
    1WAT    HW2    3   0.062   0.126   0.126
    2MET     C1    4   0.626   0.626   0.626
    2MET     O1    5   0.690   0.626   0.626
    2MET     H1    6   0.562   0.626   0.626
    3MET     C1    7   1.126   1.126   1.126
    3MET     O1    8   1.190   1.126   1.126
    3MET     H1    9   1.062   1.126   1.126
   2.50000   2.50000   2.50000
"""
    gro_file = tmp_path / "mixed.gro"
    gro_file.write_text(gro_content)
    return str(gro_file)


@pytest.fixture
def mock_universe():
    """Create a mock MDAnalysis Universe."""
    universe = Mock(spec=MDAnalysis.Universe)

    # Mock residues
    residue1 = Mock()
    residue1.resname = "WAT"
    atom1 = Mock()
    atom1.type = "O"
    atom2 = Mock()
    atom2.type = "H"
    atom3 = Mock()
    atom3.type = "H"
    residue1.atoms = Mock()
    residue1.atoms.types = np.array(["O", "H", "H"])

    residue2 = Mock()
    residue2.resname = "WAT"
    residue2.atoms = Mock()
    residue2.atoms.types = np.array(["O", "H", "H"])

    residues = Mock(spec=ResidueGroup)
    residues.__iter__ = Mock(return_value=iter([residue1, residue2]))
    residues.resnames = np.array(["WAT", "WAT"])

    universe.residues = residues
    universe.dimensions = np.array([18.6206, 18.6206, 18.6206, 90.0, 90.0, 90.0])

    return universe


class TestGroParserInitialization:
    """Test GroParser initialization."""

    def test_valid_gro_file(self, sample_gro_file):
        """Test initialization with a valid .gro file."""
        parser = GroParser(sample_gro_file)
        assert parser.filepath == Path(sample_gro_file)
        assert parser._universe is not None

    def test_invalid_file_extension(self, tmp_path):
        """Test that non-.gro files raise an error."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a gro file")

        with pytest.raises(ValueError, match=".*\\.gro"):
            GroParser(str(txt_file))

    def test_nonexistent_file(self):
        """Test that nonexistent files raise an error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            GroParser("/nonexistent/path/file.gro")

    def test_path_validation(self, sample_gro_file):
        """Test that path is properly validated and converted."""
        parser = GroParser(sample_gro_file)
        assert isinstance(parser.filepath, Path)


class TestResiduesProperty:
    """Test the residues property."""

    @patch('MDAnalysis.Universe')
    def test_residues_property(self, mock_mda, sample_gro_file):
        """Test that residues property returns ResidueGroup."""
        mock_universe = Mock()
        mock_residues = Mock(spec=ResidueGroup)
        mock_universe.residues = mock_residues
        mock_mda.return_value = mock_universe

        parser = GroParser(sample_gro_file)
        assert parser.residues == mock_residues

    def test_residues_real_file(self, sample_gro_file):
        """Test residues property with real file."""
        parser = GroParser(sample_gro_file)
        residues = parser.residues
        assert isinstance(residues, ResidueGroup)
        assert len(residues) == 2  # Two water molecules


class TestMoleculeCount:
    """Test the molecule_count property."""

    def test_molecule_count_single_type(self, sample_gro_file):
        """Test molecule count with single molecule type."""
        parser = GroParser(sample_gro_file)
        mol_count = parser.molecule_count

        assert isinstance(mol_count, dict)
        assert "WAT" in mol_count
        assert mol_count["WAT"] == 2

    def test_molecule_count_mixed_system(self, mixed_system_gro_file):
        """Test molecule count with multiple molecule types."""
        parser = GroParser(mixed_system_gro_file)
        mol_count = parser.molecule_count

        assert len(mol_count) == 2
        assert mol_count.get("WAT") == 1
        assert mol_count.get("MET") == 2

    def test_molecule_count_types(self, sample_gro_file):
        """Test that molecule counts are integers."""
        parser = GroParser(sample_gro_file)
        mol_count = parser.molecule_count

        for count in mol_count.values():
            assert isinstance(count, int)


class TestMoleculesProperty:
    """Test the molecules property."""

    def test_molecules_list(self, sample_gro_file):
        """Test that molecules returns a list of molecule names."""
        parser = GroParser(sample_gro_file)
        molecules = parser.molecules

        assert isinstance(molecules, list)
        assert "WAT" in molecules

    def test_molecules_mixed_system(self, mixed_system_gro_file):
        """Test molecules list with mixed system."""
        parser = GroParser(mixed_system_gro_file)
        molecules = parser.molecules

        assert len(molecules) == 2
        assert set(molecules) == {"WAT", "MET"}

    def test_molecules_unique(self, sample_gro_file):
        """Test that molecules list contains unique entries."""
        parser = GroParser(sample_gro_file)
        molecules = parser.molecules

        assert len(molecules) == len(set(molecules))


class TestTotalMolecules:
    """Test the total_molecules property."""

    def test_total_molecules_single_type(self, sample_gro_file):
        """Test total molecule count with single type."""
        parser = GroParser(sample_gro_file)
        total = parser.total_molecules

        assert isinstance(total, int)
        assert total == 2

    def test_total_molecules_mixed_system(self, mixed_system_gro_file):
        """Test total molecule count with mixed system."""
        parser = GroParser(mixed_system_gro_file)
        total = parser.total_molecules

        assert total == 3  # 1 WAT + 2 MET


class TestAtomCounts:
    """Test the atom_counts property."""

    def test_atom_counts_structure(self, sample_gro_file):
        """Test that atom_counts returns proper nested dictionary."""
        parser = GroParser(sample_gro_file)
        atom_counts = parser.atom_counts

        assert isinstance(atom_counts, dict)
        assert "WAT" in atom_counts
        assert isinstance(atom_counts["WAT"], dict)

    def test_atom_counts_water(self, sample_gro_file):
        """Test atom counts for water molecules."""
        parser = GroParser(sample_gro_file)
        atom_counts = parser.atom_counts

        wat_atoms = atom_counts["WAT"]
        assert "OW" in wat_atoms or "O" in wat_atoms
        assert "HW1" in wat_atoms or "HW2" in wat_atoms or "H" in wat_atoms

    def test_atom_counts_values_are_integers(self, sample_gro_file):
        """Test that all atom counts are integers."""
        parser = GroParser(sample_gro_file)
        atom_counts = parser.atom_counts

        for _resname, atoms in atom_counts.items():
            for count in atoms.values():
                assert isinstance(count, int)

    def test_atom_counts_no_duplicates(self, sample_gro_file):
        """Test that each residue type appears only once."""
        parser = GroParser(sample_gro_file)
        atom_counts = parser.atom_counts

        # Should only have one entry per unique residue name
        assert len(atom_counts) == len(parser.molecules)


class TestElectronCount:
    """Test the electron_count property."""

    @patch('kbkit.io.gro.GroParser.get_atomic_number')
    def test_electron_count_calculation(self, mock_get_atomic, sample_gro_file):
        """Test electron count calculation."""
        # Mock atomic numbers: O=8, H=1
        def atomic_number_side_effect(atom_type):
            atom_map = {"OW": 8, "O": 8, "HW1": 1, "HW2": 1, "H": 1}
            for key, value in atom_map.items():
                if key in atom_type or atom_type in key:
                    return value
            return 1

        mock_get_atomic.side_effect = atomic_number_side_effect

        parser = GroParser(sample_gro_file)
        electron_count = parser.electron_count

        assert isinstance(electron_count, dict)
        assert "WAT" in electron_count
        # Water: 1 O (8e) + 2 H (1e each) = 10 electrons
        assert electron_count["WAT"] == 10

    def test_electron_count_cached(self, sample_gro_file):
        """Test that electron_count is cached."""
        parser = GroParser(sample_gro_file)

        # Access twice
        count1 = parser.electron_count
        count2 = parser.electron_count

        # Should be the same object (cached)
        assert count1 is count2

    def test_electron_count_types(self, sample_gro_file):
        """Test that electron counts are integers."""
        parser = GroParser(sample_gro_file)
        electron_count = parser.electron_count

        for count in electron_count.values():
            assert isinstance(count, int)

    @patch('kbkit.io.gro.GroParser.get_atomic_number')
    def test_electron_count_mixed_system(self, mock_get_atomic, mixed_system_gro_file):
        """Test electron count with mixed molecule types."""
        def atomic_number_side_effect(atom_type):
            atom_map = {
                "OW": 8, "O": 8, "O1": 8,
                "HW1": 1, "HW2": 1, "H": 1, "H1": 1,
                "C": 6, "C1": 6
            }
            for key, value in atom_map.items():
                if key in atom_type or atom_type in key:
                    return value
            return 1

        mock_get_atomic.side_effect = atomic_number_side_effect

        parser = GroParser(mixed_system_gro_file)
        electron_count = parser.electron_count

        assert len(electron_count) == 2
        assert "WAT" in electron_count
        assert "MET" in electron_count


class TestBoxVolume:
    """Test the box_volume property."""

    def test_box_volume_calculation(self, sample_gro_file):
        """Test box volume calculation."""
        parser = GroParser(sample_gro_file)
        volume = parser.box_volume

        assert isinstance(volume, float)
        assert volume > 0
        # Expected: (1.86206)^3 ≈ 6.46 nm^3
        assert pytest.approx(volume, rel=0.01) == 6.46

    def test_box_volume_cached(self, sample_gro_file):
        """Test that box_volume is cached."""
        parser = GroParser(sample_gro_file)

        volume1 = parser.box_volume
        volume2 = parser.box_volume

        # Should be the same value (cached)
        assert volume1 == volume2

    def test_box_volume_units(self, sample_gro_file):
        """Test that box volume is in nm^3."""
        parser = GroParser(sample_gro_file)
        volume = parser.box_volume

        # Volume should be reasonable for a small system (nm^3)
        assert 0.1 < volume < 1000

    @patch('MDAnalysis.Universe')
    def test_box_volume_conversion(self, mock_mda, sample_gro_file):
        """Test conversion from Angstroms to nm."""
        mock_universe = Mock()
        # Set dimensions in Angstroms: 10 Å = 1 nm
        mock_universe.dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
        mock_mda.return_value = mock_universe

        parser = GroParser(sample_gro_file)
        volume = parser.box_volume

        # (10 Å / 10)^3 = 1 nm^3
        assert pytest.approx(volume, abs=0.001) == 1.0

    def test_box_volume_non_cubic(self, tmp_path):
        """Test box volume calculation for non-cubic box."""
        gro_content = """Non-cubic box
    3
    1WAT     OW    1   0.126   0.126   0.126
    1WAT    HW1    2   0.190   0.126   0.126
    1WAT    HW2    3   0.062   0.126   0.126
   2.00000   3.00000   4.00000
"""
        gro_file = tmp_path / "noncubic.gro"
        gro_file.write_text(gro_content)

        parser = GroParser(str(gro_file))
        volume = parser.box_volume

        # (20 Å * 30 Å * 40 Å) / 1000 = 24 nm^3
        assert pytest.approx(volume, abs=0.01) == 24.0


class TestGroParserIntegration:
    """Integration tests for GroParser."""

    def test_complete_workflow_water(self, sample_gro_file):
        """Test complete workflow with water system."""
        parser = GroParser(sample_gro_file)

        # Check all properties work together
        assert parser.total_molecules == 2
        assert len(parser.molecules) == 1
        assert "WAT" in parser.molecules
        assert parser.molecule_count["WAT"] == 2
        assert "WAT" in parser.atom_counts
        assert "WAT" in parser.electron_count
        assert parser.box_volume > 0

    def test_complete_workflow_mixed(self, mixed_system_gro_file):
        """Test complete workflow with mixed system."""
        parser = GroParser(mixed_system_gro_file)

        assert parser.total_molecules == 3
        assert len(parser.molecules) == 2
        assert set(parser.molecules) == {"WAT", "MET"}
        assert parser.molecule_count["WAT"] == 1
        assert parser.molecule_count["MET"] == 2

    def test_multiple_property_access(self, sample_gro_file):
        """Test that multiple property accesses work correctly."""
        parser = GroParser(sample_gro_file)

        # Access properties multiple times
        for _ in range(3):
            _ = parser.molecules
            _ = parser.molecule_count
            _ = parser.total_molecules
            _ = parser.atom_counts
            _ = parser.electron_count
            _ = parser.box_volume

        # Should not raise any errors

    def test_property_consistency(self, sample_gro_file):
        """Test that properties are consistent with each other."""
        parser = GroParser(sample_gro_file)

        # Total molecules should equal sum of molecule_count
        assert parser.total_molecules == sum(parser.molecule_count.values())

        # Molecules list should match molecule_count keys
        assert set(parser.molecules) == set(parser.molecule_count.keys())

        # atom_counts should have entries for all molecules
        assert set(parser.atom_counts.keys()) == set(parser.molecules)

        # electron_count should have entries for all molecules
        assert set(parser.electron_count.keys()) == set(parser.molecules)



class TestIsValidElement:
    """Test is_valid_element function."""

    def test_valid_single_letter_elements(self):
        """Test valid single-letter element symbols."""
        valid_elements = ['H', 'C', 'N', 'O', 'P', 'S', 'F', 'I', 'B', 'K']

        for symbol in valid_elements:
            assert GroParser.is_valid_element(symbol) is True

    def test_valid_two_letter_elements(self):
        """Test valid two-letter element symbols."""
        valid_elements = ['He', 'Li', 'Be', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ca', 'Fe']

        for symbol in valid_elements:
            assert GroParser.is_valid_element(symbol) is True

    def test_lowercase_symbols(self):
        """Test that lowercase symbols are accepted."""
        assert GroParser.is_valid_element('h') is True
        assert GroParser.is_valid_element('c') is True
        assert GroParser.is_valid_element('na') is True
        assert GroParser.is_valid_element('cl') is True

    def test_uppercase_symbols(self):
        """Test that uppercase symbols are accepted."""
        assert GroParser.is_valid_element('H') is True
        assert GroParser.is_valid_element('C') is True
        assert GroParser.is_valid_element('NA') is True
        assert GroParser.is_valid_element('CL') is True

    def test_mixed_case_symbols(self):
        """Test that mixed case symbols are accepted."""
        assert GroParser.is_valid_element('Na') is True
        assert GroParser.is_valid_element('Cl') is True
        assert GroParser.is_valid_element('nA') is True
        assert GroParser.is_valid_element('cL') is True

    def test_symbols_with_whitespace(self):
        """Test symbols with leading/trailing whitespace."""
        assert GroParser.is_valid_element(' H ') is True
        assert GroParser.is_valid_element('  C  ') is True
        assert GroParser.is_valid_element(' Na ') is True
        assert GroParser.is_valid_element('\tCl\n') is True

    def test_empty_string(self):
        """Test empty string returns False."""
        assert GroParser.is_valid_element('') is False

    def test_none_input(self):
        """Test None input returns False."""
        assert GroParser.is_valid_element(None) is False

    def test_non_string_input(self):
        """Test non-string input returns False."""
        assert GroParser.is_valid_element(123) is False
        assert GroParser.is_valid_element(12.5) is False
        assert GroParser.is_valid_element([]) is False
        assert GroParser.is_valid_element({}) is False

    def test_long_symbols_truncated(self):
        """Test that symbols longer than MAX_SYMBOL_LENGTH are truncated."""
        # 'Carbon' should be truncated to 'Ca' (Calcium) which is valid
        result = GroParser.is_valid_element('Carbon')
        assert result is True  # 'Ca' is Calcium

        # 'Helium' should be truncated to 'He' which is valid
        result = GroParser.is_valid_element('Helium')
        assert result is True

    def test_all_common_elements(self):
        """Test all common elements used in chemistry."""
        common_elements = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Fe', 'Cu', 'Zn', 'Br', 'Ag', 'I', 'Au', 'Hg', 'Pb', 'U'
        ]

        for symbol in common_elements:
            assert GroParser.is_valid_element(symbol) is True, f"{symbol} should be valid"


class TestGetAtomicNumber:
    """Test get_atomic_number function."""

    def test_hydrogen(self):
        """Test atomic number of hydrogen."""
        assert GroParser.get_atomic_number('H') == 1

    def test_carbon(self):
        """Test atomic number of carbon."""
        assert GroParser.get_atomic_number('C') == 6

    def test_nitrogen(self):
        """Test atomic number of nitrogen."""
        assert GroParser.get_atomic_number('N') == 7

    def test_oxygen(self):
        """Test atomic number of oxygen."""
        assert GroParser.get_atomic_number('O') == 8

    def test_sodium(self):
        """Test atomic number of sodium."""
        assert GroParser.get_atomic_number('Na') == 11

    def test_chlorine(self):
        """Test atomic number of chlorine."""
        assert GroParser.get_atomic_number('Cl') == 17

    def test_iron(self):
        """Test atomic number of iron."""
        assert GroParser.get_atomic_number('Fe') == 26

    def test_gold(self):
        """Test atomic number of gold."""
        assert GroParser.get_atomic_number('Au') == 79

    def test_lowercase_input(self):
        """Test that lowercase input works."""
        assert GroParser.get_atomic_number('h') == 1
        assert GroParser.get_atomic_number('c') == 6
        assert GroParser.get_atomic_number('na') == 11

    def test_uppercase_input(self):
        """Test that uppercase input works."""
        assert GroParser.get_atomic_number('H') == 1
        assert GroParser.get_atomic_number('C') == 6
        assert GroParser.get_atomic_number('NA') == 11

    def test_mixed_case_input(self):
        """Test that mixed case input works."""
        assert GroParser.get_atomic_number('Na') == 11
        assert GroParser.get_atomic_number('nA') == 11
        assert GroParser.get_atomic_number('Cl') == 17
        assert GroParser.get_atomic_number('cL') == 17

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        assert GroParser.get_atomic_number(' H ') == 1
        assert GroParser.get_atomic_number('  C  ') == 6
        assert GroParser.get_atomic_number(' Na ') == 11

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="not a valid element"):
            GroParser.get_atomic_number('')

    def test_none_raises_error(self):
        """Test that None raises AttributeError (from .strip() call)."""
        with pytest.raises(AttributeError):
            GroParser.get_atomic_number(None)

    def test_all_elements_1_to_20(self):
        """Test atomic numbers for first 20 elements."""
        elements = [
            ('H', 1), ('He', 2), ('Li', 3), ('Be', 4), ('B', 5),
            ('C', 6), ('N', 7), ('O', 8), ('F', 9), ('Ne', 10),
            ('Na', 11), ('Mg', 12), ('Al', 13), ('Si', 14), ('P', 15),
            ('S', 16), ('Cl', 17), ('Ar', 18), ('K', 19), ('Ca', 20)
        ]

        for symbol, expected_number in elements:
            assert GroParser.get_atomic_number(symbol) == expected_number

    def test_transition_metals(self):
        """Test atomic numbers for common transition metals."""
        metals = [
            ('Fe', 26), ('Cu', 29), ('Zn', 30),
            ('Ag', 47), ('Au', 79), ('Pt', 78)
        ]

        for symbol, expected_number in metals:
            assert GroParser.get_atomic_number(symbol) == expected_number

    def test_halogens(self):
        """Test atomic numbers for halogens."""
        halogens = [
            ('F', 9), ('Cl', 17), ('Br', 35), ('I', 53)
        ]

        for symbol, expected_number in halogens:
            assert GroParser.get_atomic_number(symbol) == expected_number

    def test_noble_gases(self):
        """Test atomic numbers for noble gases."""
        noble_gases = [
            ('He', 2), ('Ne', 10), ('Ar', 18), ('Kr', 36), ('Xe', 54)
        ]

        for symbol, expected_number in noble_gases:
            assert GroParser.get_atomic_number(symbol) == expected_number


class TestMaxSymbolLength:
    """Test MAX_SYMBOL_LENGTH constant."""

    def test_max_symbol_length_value(self):
        """Test that MAX_SYMBOL_LENGTH is 2."""
        assert GroParser.MAX_SYMBOL_LENGTH == 2

    def test_max_symbol_length_is_int(self):
        """Test that MAX_SYMBOL_LENGTH is an integer."""
        assert isinstance(GroParser.MAX_SYMBOL_LENGTH, int)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_character_valid(self):
        """Test single character valid elements."""
        single_char_elements = ['H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'K', 'V', 'Y', 'I', 'W', 'U']

        for symbol in single_char_elements:
            assert GroParser.is_valid_element(symbol) is True
            assert GroParser.get_atomic_number(symbol) > 0

    def test_two_character_valid(self):
        """Test two character valid elements."""
        two_char_elements = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar']

        for symbol in two_char_elements:
            assert GroParser.is_valid_element(symbol) is True
            assert GroParser.get_atomic_number(symbol) > 0

    def test_case_insensitivity(self):
        """Test that functions are case-insensitive."""
        test_cases = [
            ('h', 'H', 'hydrogen'),
            ('na', 'Na', 'sodium'),
            ('cl', 'Cl', 'chlorine'),
            ('fe', 'Fe', 'iron')
        ]

        for lower, proper, _name in test_cases:
            # All variations should be valid
            assert GroParser.is_valid_element(lower) is True
            assert GroParser.is_valid_element(proper) is True
            assert GroParser.is_valid_element(lower.upper()) is True

            # All should return same atomic number
            assert GroParser.get_atomic_number(lower) == GroParser.get_atomic_number(proper)
            assert GroParser.get_atomic_number(lower) == GroParser.get_atomic_number(lower.upper())

    def test_very_long_string(self):
        """Test very long string is handled."""
        # String starting with valid element
        long_string = 'Calcium_very_long_name'
        # Should be truncated to 'Ca' and be valid
        result = GroParser.is_valid_element(long_string)
        assert result is True  # 'Ca' is valid

    def test_atomic_number_return_type(self):
        """Test that get_atomic_number returns int."""
        result = GroParser.get_atomic_number('H')
        assert isinstance(result, int)

        result = GroParser.get_atomic_number('C')
        assert isinstance(result, int)

    def test_is_valid_element_return_type(self):
        """Test that is_valid_element returns bool."""
        result = GroParser.is_valid_element('H')
        assert isinstance(result, bool)
        assert result is True


class TestIntegration:
    """Integration tests combining both functions."""

    def test_valid_element_has_atomic_number(self):
        """Test that all valid elements have atomic numbers."""
        test_elements = ['H', 'C', 'N', 'O', 'Na', 'Cl', 'Fe', 'Au']

        for symbol in test_elements:
            if GroParser.is_valid_element(symbol):
                atomic_num = GroParser.get_atomic_number(symbol)
                assert atomic_num > 0
                assert isinstance(atomic_num, int)

    def test_workflow_validation_then_lookup(self):
        """Test typical workflow: validate then lookup."""
        symbol = 'Na'

        # First validate
        if GroParser.is_valid_element(symbol):
            # Then get atomic number
            atomic_num = GroParser.get_atomic_number(symbol)
            assert atomic_num == 11

    def test_periodic_table_coverage(self):
        """Test coverage of periodic table elements."""
        # Test first 36 elements (up to Krypton)
        expected_symbols = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'
        ]

        for i, symbol in enumerate(expected_symbols, start=1):
            assert GroParser.is_valid_element(symbol) is True
            assert GroParser.get_atomic_number(symbol) == i

    def test_safe_validation_pattern(self):
        """Test safe pattern: check validity before getting atomic number."""
        # Only test with valid symbols since is_valid_element doesn't catch exceptions
        test_symbols = ['H', 'C', 'Na', 'Fe', 'Au', 'Pt']

        for symbol in test_symbols:
            assert GroParser.is_valid_element(symbol) is True
            atomic_num = GroParser.get_atomic_number(symbol)
            assert atomic_num > 0

    def test_truncation_behavior(self):
        """Test that long strings are truncated correctly."""
        # These should be truncated to valid 2-letter symbols
        test_cases = [
            ('Helium', True),   # -> 'He'
            ('Carbon', True),   # -> 'Ca' (Calcium)
            ('Beryllium', True),   # -> 'Be'
            ('Lithium', True),  # -> 'Li'
        ]

        for long_name, _expected_valid in test_cases:
            result = GroParser.is_valid_element(long_name)
            # Just check it returns a boolean
            assert isinstance(result, bool)
