import sys
from pathlib import Path
import subprocess

def main():
    """Launch the Windows 11 menu installer with proper UI feedback"""
    script_dir = Path(__file__).parent
    installer_path = script_dir / "win11_menu_installer.py"
    
    print("=== PNG Metadata Viewer Installation ===")
    print(f"Installing from: {script_dir.absolute()}")
    
    try:
        # Run the installer with an argument to suppress its own exit prompt
        if len(sys.argv) > 1 and sys.argv[1] == "--uninstall":
            print("\nUninstalling PNG Metadata Viewer...")
            subprocess.run([sys.executable, str(installer_path), "--uninstall", "--no-prompt"])
        else:
            print("\nInstalling PNG Metadata Viewer...")
            subprocess.run([sys.executable, str(installer_path), "--no-prompt"])
            
        print("\nAll done!")
    except Exception as e:
        print(f"\nError during installation: {e}")
        print("Please try running as administrator if you encounter permission issues.")
    
    # Single prompt to exit
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()