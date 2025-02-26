"""
Test suite for PNG metadata handling with proper British standards
"""
import pytest
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json
import numpy as np
from png_metadata_tools.png_inspector import EnhancedPNGInspector

# Test data paths
TEST_ROOT = Path(__file__).parent
TEST_DATA = TEST_ROOT / "data"
TEST_IMAGES = TEST_DATA / "images"
TEST_CASES = TEST_IMAGES / "test_cases"
TEST_TEMP = TEST_DATA / "temp"

def create_test_image(path: Path, size=(512, 512), metadata=None, elo_rating=None) -> None:
    """Helper function to create test images with metadata"""
    # Create image with some variation
    img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Add metadata
    meta = PngInfo()
    if metadata:
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                meta.add_text(key, json.dumps(value))
            else:
                meta.add_text(key, str(value))
    if elo_rating is not None:
        meta.add_text('elo_rating', str(elo_rating))
    
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, pnginfo=meta)

class TestPNGMetadata:
    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Setup test environment and generate test files"""
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        TEST_CASES.mkdir(parents=True, exist_ok=True)
        
        # Create test files
        create_test_image(TEST_CASES / "british_with_rating.png", elo_rating=1500.0)
        create_test_image(TEST_CASES / "british_with_invalid_rating.png", elo_rating="not_a_number")
        create_test_image(TEST_CASES / "empty_metadata.png")
        create_test_image(TEST_CASES / "only_rating.png", elo_rating=1200.0)
        
        # Create mixed metadata test file
        mixed_metadata = {
            'parameters': {
                'prompt': 'test prompt',
                'seed': 12345,
                'steps': 20,
                'cfg': 7.5,
                'sampler': 'euler_a',
            }
        }
        create_test_image(TEST_CASES / "mixed_metadata.png", metadata=mixed_metadata, elo_rating=1500.0)
        
        # Create partial ComfyUI metadata test file
        partial_metadata = {'parameters': "invalid json"}
        create_test_image(TEST_CASES / "partial_comfy.png", metadata=partial_metadata)
        
        yield
        
        # Cleanup
        for file in TEST_TEMP.glob("*"):
            file.unlink(missing_ok=True)

    def read_metadata(self, image_path: Path) -> dict:
        """Use EnhancedPNGInspector to read metadata"""
        inspector = EnhancedPNGInspector(str(image_path))
        return inspector.metadata

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
        assert metadata['elo_rating'] == "Found but invalid format"

    def test_empty_metadata(self):
        """Test image with no metadata"""
        metadata = self.read_metadata(TEST_CASES / "empty_metadata.png")
        assert not metadata.get('text_keys', [])

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
        assert 'parameters' in metadata['raw_text']
        assert isinstance(metadata['elo_rating'], float)

    def test_partial_comfy(self):
        """Test image with partial ComfyUI metadata"""
        metadata = self.read_metadata(TEST_CASES / "partial_comfy.png")
        assert 'text' in metadata
        assert 'parameters' in metadata['raw_text']

    def test_image_data_integrity(self):
        """Test that image data remains intact after metadata operations"""
        test_files = [
            "british_with_rating.png",
            "british_with_invalid_rating.png"
        ]
        
        for test_file in test_files:
            with Image.open(TEST_CASES / test_file) as img:
                img.verify()  # Should not raise an exception

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
            if 'elo_rating' in control_result:
                assert control_result['elo_rating'] == implementation_result.get('elo_rating'), \
                    f"Mismatch in elo_rating for {test_file}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])