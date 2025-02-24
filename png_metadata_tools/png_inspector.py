"""
Enhanced PNG Metadata Inspector with detailed ComfyUI metadata parsing
"""
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json
from pathlib import Path
from typing import Dict, Any, Optional
from pprint import pprint

class EnhancedPNGInspector:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.metadata: Dict[str, Any] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Extract all metadata from PNG file with detailed parsing."""
        try:
            with Image.open(self.image_path) as img:
                # Get basic image info
                self.metadata['size'] = img.size
                self.metadata['mode'] = img.mode
                
                # Get all text metadata
                if hasattr(img, 'text'):
                    self.metadata['text'] = dict(img.text)  # Store original text
                    self.metadata['raw_text'] = dict(img.text)  # Keep raw_text for backwards compatibility
                    self.metadata['text_keys'] = list(img.text.keys())
                    
                    # Parse ComfyUI parameters if present
                    if 'parameters' in img.text:
                        try:
                            params = json.loads(img.text['parameters'])
                            self.metadata['comfy_parameters'] = params
                        except json.JSONDecodeError:
                            self.metadata['comfy_parameters'] = "Found but failed to parse"
                    
                    # Look for rating data
                    if 'elo_rating' in img.text:
                        try:
                            self.metadata['elo_rating'] = float(img.text['elo_rating'])
                        except ValueError:
                            self.metadata['elo_rating'] = "Found but invalid format"
                else:
                    # Ensure text field exists even if empty
                    self.metadata['text'] = {}
                    self.metadata['raw_text'] = {}
                    self.metadata['text_keys'] = []
                    
        except FileNotFoundError:
            print(f"Error: File not found - {self.image_path}")
            self.metadata['error'] = 'File not found'
            self.metadata['text'] = {}
            self.metadata['raw_text'] = {}
        except Image.UnidentifiedImageError:
            print(f"Error: Not a valid PNG file or corrupted - {self.image_path}")
            self.metadata['error'] = 'Invalid or corrupted PNG'
            self.metadata['text'] = {}
            self.metadata['raw_text'] = {}
        except Exception as e:
            print(f"Error: Unexpected error while reading {self.image_path} - {str(e)}")
            self.metadata['error'] = f'Unexpected error: {str(e)}'
            self.metadata['text'] = {}
            self.metadata['raw_text'] = {}

    def print_detailed_summary(self) -> None:
        """Print a detailed formatted summary of the metadata."""
        print(f"\n{'='*80}")
        print(f"Detailed Metadata Summary for {Path(self.image_path).name}")
        print(f"{'='*80}")
        
        if 'error' in self.metadata:
            print(f"\nERROR ENCOUNTERED:")
            print(self.metadata['error'])
            return

        print(f"\nBASIC INFORMATION:")
        print(f"Image Size: {self.metadata['size']}")
        print(f"Image Mode: {self.metadata['mode']}")
        
        if 'text_keys' in self.metadata:
            print(f"\nMETADATA KEYS FOUND: {', '.join(self.metadata['text_keys'])}")
        
        if 'comfy_parameters' in self.metadata:
            print("\nCOMFYUI PARAMETERS:")
            if isinstance(self.metadata['comfy_parameters'], dict):
                pprint(self.metadata['comfy_parameters'], indent=2)
            else:
                print(self.metadata['comfy_parameters'])
        
        if 'elo_rating' in self.metadata:
            print(f"\nELO RATING: {self.metadata['elo_rating']}")
        
        print("\nRAW METADATA:")
        if 'raw_text' in self.metadata:
            pprint(self.metadata['raw_text'], indent=2)

def inspect_file(filepath: str) -> None:
    """Inspect a single PNG file."""
    inspector = EnhancedPNGInspector(filepath)
    inspector.print_detailed_summary()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python enhanced_inspector.py <image_path>")
        sys.exit(1)
    
    inspect_file(sys.argv[1])