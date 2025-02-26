"""
A proper British utility for dumping project contents with PNG validation.
"""
import os
from pathlib import Path
from PIL import Image
import hashlib
from datetime import datetime

class ProperBritishSourceDumper:
    # Directories that shan't be included
    SKIP_DIRS = {
        'node_modules',
        '__pycache__',
        '.git',
        'build',
        'dist',
        'venv',
        '.next',
        'coverage',
        '.pytest_cache',
        'temp'
    }

    # Files that shan't be included
    SKIP_FILES = {
        'package-lock.json',
        'yarn.lock',
        'pnpm-lock.yaml',
        'poetry.lock',
        'Pipfile.lock',
        'src_contents.txt',
        'dump_src.py'
    }

    # Extensions we shall process
    INCLUDE_EXTENSIONS = {
        # Python files
        '.py',
        # Configuration files
        '.json', '.toml', '.yml', '.yaml',
        # Documentation
        '.md', '.rst',
        # Test data (excluding PNGs, which get special treatment)
        '.txt'
    }

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.png_manifest = []

    def _validate_png(self, file_path: Path) -> dict:
        """Examine a PNG file for proper British standards."""
        try:
            with Image.open(file_path) as img:
                metadata = {
                    'filename': file_path.name,
                    'size': img.size,
                    'mode': img.mode,
                    'has_metadata': bool(getattr(img, 'text', None)),
                }
                # Calculate a proper checksum
                with open(file_path, 'rb') as f:
                    metadata['checksum'] = hashlib.sha256(f.read()).hexdigest()[:8]
                return metadata
        except Exception as e:
            return {
                'filename': file_path.name,
                'error': str(e)
            }

    def dump_contents(self, output_file: str = "src_contents.txt"):
        """
        Dump the project contents while maintaining proper British standards.
        """
        timestamp = datetime.now().strftime("%d %B %Y, %H:%M")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write a proper header
            f.write(f"Project Source Contents - Generated on {timestamp}\n")
            f.write("=" * 80 + "\n\n")

            # Process all files
            for root, dirs, files in os.walk(self.project_root):
                # Skip unwanted directories with proper manners
                dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]
                
                # Process each file with dignity
                for file in files:
                    if file in self.SKIP_FILES:
                        continue
                        
                    file_path = Path(root) / file
                    try:
                        # Handle PNGs with special care
                        if file_path.suffix.lower() == '.png':
                            png_info = self._validate_png(file_path)
                            self.png_manifest.append(png_info)
                            continue

                        # Process other acceptable files
                        if file_path.suffix in self.INCLUDE_EXTENSIONS:
                            rel_path = file_path.relative_to(self.project_root)
                            
                            # Mind the file size, like a proper gentleman
                            if file_path.stat().st_size > 1_000_000:
                                print(f"I say, this file is rather large: {file_path}")
                                continue
                            
                            f.write(f"\n{'='*80}\n")
                            f.write(f"FILE: {rel_path}\n")
                            f.write(f"{'='*80}\n\n")
                            
                            try:
                                with open(file_path, 'r', encoding='utf-8') as source_file:
                                    f.write(source_file.read())
                            except Exception as e:
                                f.write(f"My sincerest apologies, but there was an error reading this file: {e}\n")
                            
                            f.write("\n\n")
                            
                    except Exception as e:
                        print(f"Do pardon me, but there was an error processing {file_path}: {e}")

            # Append the PNG manifest with proper formatting
            if self.png_manifest:
                f.write("\nPNG Assets Manifest\n")
                f.write("=" * 80 + "\n\n")
                for png in self.png_manifest:
                    f.write("PNG File: {}\n".format(png['filename']))
                    for key, value in png.items():
                        if key != 'filename':
                            f.write(f"  {key}: {value}\n")
                    f.write("\n")

if __name__ == "__main__":
    try:
        dumper = ProperBritishSourceDumper()
        dumper.dump_contents()
        print("I'm pleased to report that the contents have been successfully documented.")
    except Exception as e:
        print(f"I do apologize, but an error has occurred: {e}")