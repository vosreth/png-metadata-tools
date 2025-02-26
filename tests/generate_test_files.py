"""
Enhanced Test File Generator with proper British standards support
"""
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json
from pathlib import Path
import shutil

class TestImageGenerator:
    def __init__(self, output_dir: Path, comfy_image_path: Path = None):
        self.base_dir = Path(__file__).parent
        self.output_dir = self.base_dir / "data/images"
        self.test_cases_dir = self.output_dir / "test_cases"
        self.comfy_image_path = comfy_image_path
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_cases_dir.mkdir(parents=True, exist_ok=True)
        
        # Reference to our British standards image
        self.british_standards_image = self.output_dir / "proper_british_standards.png"

    def get_comfy_metadata(self) -> tuple[dict, dict]:
        """Extract ComfyUI metadata from source image"""
        if not self.comfy_image_path or not self.comfy_image_path.exists():
            return None, None
            
        with Image.open(self.comfy_image_path) as img:
            if not hasattr(img, 'text'):
                return None, None
                
            prompt = json.loads(img.text['prompt']) if 'prompt' in img.text else None
            workflow = json.loads(img.text['workflow']) if 'workflow' in img.text else None
            
            return prompt, workflow

    def create_image_with_metadata(self, path: Path, base_image: Image.Image | Path, 
                                 comfy_metadata: dict = None, elo_rating: float = None) -> None:
        """Create a test image with specified metadata"""
        metadata = PngInfo()
        
        # Add ComfyUI metadata if provided
        if comfy_metadata and 'prompt' in comfy_metadata and 'workflow' in comfy_metadata:
            metadata.add_text('prompt', json.dumps(comfy_metadata['prompt']))
            metadata.add_text('workflow', json.dumps(comfy_metadata['workflow']))
            
        # Add elo_rating if provided
        if elo_rating is not None:
            metadata.add_text('elo_rating', str(elo_rating))
            
        # Save with metadata
        if isinstance(base_image, Image.Image):
            base_image.save(path, 'PNG', pnginfo=metadata)
        else:
            with Image.open(base_image) as img:
                img.save(path, 'PNG', pnginfo=metadata)

    def generate_test_files(self):
        """Generate test files using authentic ComfyUI metadata and British standards image"""
        # Get ComfyUI metadata
        comfy_metadata = None
        if self.comfy_image_path and self.comfy_image_path.exists():
            prompt, workflow = self.get_comfy_metadata()
            if prompt and workflow:
                comfy_metadata = {'prompt': prompt, 'workflow': workflow}

        # Generate test cases using British standards image
        if self.british_standards_image.exists():
            # 1. British standards with rating
            self.create_image_with_metadata(
                self.test_cases_dir / "british_with_rating.png",
                self.british_standards_image,
                elo_rating=1500.0
            )

            # 2. British standards with ComfyUI metadata and rating
            if comfy_metadata:
                self.create_image_with_metadata(
                    self.test_cases_dir / "british_with_comfy_and_rating.png",
                    self.british_standards_image,
                    comfy_metadata,
                    elo_rating=1500.0
                )

            # 3. British standards with invalid rating
            self.create_image_with_metadata(
                self.test_cases_dir / "british_with_invalid_rating.png",
                self.british_standards_image,
                elo_rating="not_a_number"
            )

        # Create small test cases
        base_image = Image.new('RGB', (512, 512), (240, 240, 240))

        # 1. Mixed metadata case
        mixed_metadata = {
            'elo_rating': '1500.0',
            'parameters': {
                'prompt': 'test prompt',
                'seed': 12345,
                'steps': 20,
                'cfg': 7.5,
                'sampler': 'euler_a',
            }
        }
        self.create_image_with_metadata(
            self.test_cases_dir / "mixed_metadata.png",
            base_image,
            mixed_metadata
        )
        
        # 2. Partial ComfyUI metadata case
        partial_metadata = {'parameters': {"incomplete": "data"}}  # Intentionally malformed
        self.create_image_with_metadata(
            self.test_cases_dir / "partial_comfy.png",
            base_image,
            partial_metadata
        )

        # 1. Empty metadata
        self.create_image_with_metadata(
            self.test_cases_dir / "empty_metadata.png",
            base_image
        )

        # 2. Only rating
        self.create_image_with_metadata(
            self.test_cases_dir / "only_rating.png",
            base_image,
            elo_rating=1200.0
        )

        print("\nTest files generated successfully!")
        print("\nTest cases available:")
        for file in self.test_cases_dir.glob("*.png"):
            print(f"- {file.name}")

def main():
    # Generate test files using both ComfyUI image and British standards image
    generator = TestImageGenerator(
        output_dir=Path("tests/data/images"),
        comfy_image_path=Path("../image_handler/comfyui_image.png")
    )
    generator.generate_test_files()

if __name__ == "__main__":
    main()