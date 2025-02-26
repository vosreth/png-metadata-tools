import sys
import os
import winreg
import subprocess
from pathlib import Path

# Add this near the top, right after the imports
show_prompt = True
if "--no-prompt" in sys.argv:
    show_prompt = False
    # Remove this argument to not interfere with other argument processing
    sys.argv.remove("--no-prompt")

def install_windows11_context_menu():
    """Install a Windows 11 compatible context menu for PNG files"""
    try:
        # Get path to this script
        script_path = Path(__file__).resolve()
        script_dir = script_path.parent
        
        # Path to the viewer script
        viewer_script = script_dir / "png_metadata_viewer.py"
        
        # Create batch file launcher
        batch_file = script_dir / "launch_viewer.bat"
        
        # Get pythonw.exe path - use same directory as python.exe but with 'w' suffix
        python_path = Path(sys.executable)
        pythonw_path = python_path.parent / f"{python_path.stem}w{python_path.suffix}"
        
        batch_content = f"""@echo off
start "" /b "{pythonw_path}" "{viewer_script}" "%1"
"""
        with open(batch_file, 'w') as f:
            f.write(batch_content)
        
        # Create registry key for all PNG files
        key_path = r"Software\Classes\SystemFileAssociations\.png\shell\ViewPNGMetadata"
        
        # Create the main command key
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "View PNG Metadata")
            # Set an icon (optional)
            winreg.SetValueEx(key, "Icon", 0, winreg.REG_SZ, "shell32.dll,13")
        
        # Create command subkey
        command_key_path = f"{key_path}\\command"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, command_key_path) as key:
            command = f'"{batch_file}" "%1"'
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, command)
        
        print("Context menu successfully installed!")
        return True
        
    except Exception as e:
        print(f"Error installing context menu: {e}")
        return False

def uninstall_windows11_context_menu():
    """Remove the Windows 11 context menu entry"""
    try:
        key_path = r"Software\Classes\SystemFileAssociations\.png\shell\ViewPNGMetadata"
        # Delete the key and all subkeys
        subprocess.run(["reg", "delete", f"HKCU\\{key_path}", "/f"], 
                      stderr=subprocess.DEVNULL)
        print("Context menu successfully removed!")
        return True
        
    except Exception as e:
        print(f"Error removing context menu: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--uninstall":
        uninstall_windows11_context_menu()
    else:
        install_windows11_context_menu()
    
    # Replace the original input call with this conditional version
    if show_prompt:
        input("Press Enter to exit...")