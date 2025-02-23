"""
Test suite for PNG metadata handling with proper British standards
"""
import pytest
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json

# Test data paths
TEST_ROOT = Path(__file__).parent
TEST_DATA = TEST_ROOT / "data"
TEST_IMAGES = TEST_DATA / "images"
TEST_CASES = TEST_IMAGES / "test_cases"
TEST_TEMP = TEST_DATA / "temp"

class TestPNGMetadata:
    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Setup test environment"""
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        yield
        # Cleanup temporary files after tests
        for file in TEST_TEMP.glob("*"):
            file.unlink(missing_ok=True)

    def _read_metadata_control(self, image_path: Path) -> dict:
        """
        Control implementation for metadata reading.
        Used to verify the behavior of the main implementation.
        """
        with Image.open(image_path) as img:
            metadata = {}
            if hasattr(img, 'text'):
                metadata['text'] = dict(img.text)
                if 'elo_rating' in img.text:
                    try:
                        metadata['elo_rating'] = float(img.text['elo_rating'])
                    except ValueError:
                        metadata['elo_rating'] = None
                if 'prompt' in img.text:
                    try:
                        metadata['prompt'] = json.loads(img.text['prompt'])
                    except json.JSONDecodeError:
                        metadata['prompt'] = None
            return metadata

    def test_british_standards_with_rating(self):
        """Test British standards image with proper elo rating"""
        metadata = self.read_metadata(TEST_CASES / "british_with_rating.png")
        assert 'elo_rating' in metadata
        assert isinstance(metadata['elo_rating'], float)
        assert metadata['elo_rating'] == 1500.0

    def test_british_standards_invalid_rating(self):
        """Test British standards image with invalid rating"""
        metadata = self.read_metadata(TEST_CASES / "british_with_invalid_rating.png")
        assert 'text' in metadata
        assert 'elo_rating' in metadata['text']
        assert metadata['elo_rating'] is None  # Should fail float conversion

    def test_empty_metadata(self):
        """Test image with no metadata"""
        metadata = self.read_metadata(TEST_CASES / "empty_metadata.png")
        assert not metadata.get('text', {})

    def test_only_rating(self):
        """Test image with only elo rating"""
        metadata = self.read_metadata(TEST_CASES / "only_rating.png")
        assert 'elo_rating' in metadata
        assert isinstance(metadata['elo_rating'], float)
        assert metadata['elo_rating'] == 1200.0

    def test_mixed_metadata(self):
        """Test image with mixed metadata types"""
        metadata = self.read_metadata(TEST_CASES / "mixed_metadata.png")
        assert 'elo_rating' in metadata
        assert 'prompt' in metadata
        assert isinstance(metadata['elo_rating'], float)
        assert metadata['prompt'] is not None

    def test_partial_comfy(self):
        """Test image with partial ComfyUI metadata"""
        metadata = self.read_metadata(TEST_CASES / "partial_comfy.png")
        assert 'text' in metadata
        assert 'prompt' in metadata['text']
        assert metadata['prompt'] is None  # Should fail JSON parsing

    def test_image_data_integrity(self):
        """Test that image data remains intact after metadata operations"""
        # Read original British standards image
        with Image.open(TEST_IMAGES / "proper_british_standards.png") as orig_img:
            orig_data = orig_img.tobytes()

        # Compare with metadata-modified versions
        test_files = [
            "british_with_rating.png",
            "british_with_invalid_rating.png"
        ]
        
        for test_file in test_files:
            with Image.open(TEST_CASES / test_file) as test_img:
                assert test_img.tobytes() == orig_data, f"Image data changed in {test_file}"

    @pytest.mark.parametrize("test_file", [
        "british_with_rating.png",
        "empty_metadata.png",
        "only_rating.png",
        "mixed_metadata.png",
        "partial_comfy.png"
    ])
    def test_file_accessibility(self, test_file):
        """Test that all test files are accessible and valid PNGs"""
        test_path = TEST_CASES / test_file
        assert test_path.exists(), f"Test file {test_file} not found"
        with Image.open(test_path) as img:
            assert img.format == "PNG", f"{test_file} is not a valid PNG"

    def test_metadata_reading_implementation_matches_control(self):
        """Verify that our implementation matches the control implementation"""
        test_files = [
            "british_with_rating.png",
            "empty_metadata.png",
            "mixed_metadata.png"
        ]
        
        for test_file in test_files:
            path = TEST_CASES / test_file
            control_result = self._read_metadata_control(path)
            inspector = EnhancedPNGInspector(str(path))
            implementation_result = inspector.metadata
            
            # Compare relevant fields
            assert control_result.get('elo_rating') == implementation_result.get('elo_rating'), \
                f"Mismatch in elo_rating for {test_file}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])