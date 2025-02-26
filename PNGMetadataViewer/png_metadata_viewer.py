# png_metadata_viewer.py - Enhanced with Search (Compatible)

import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import json

# Add parent directory to path to find our package
sys.path.append(str(Path(__file__).parent.parent))

# Import our library
try:
    import png_metadata_tools as pngmeta
except ImportError:
    print("PNG Metadata Tools not found. Please ensure the library is installed.")
    sys.exit(1)

class PNGMetadataViewer:
    def __init__(self, root):
        self.root = root
        self.current_file = None
        self.metadata = None
        self.all_paths = []  # Store all metadata paths for search
        self.search_matches = []  # Store search results
        self.current_match = -1  # Current position in search results
        self.setup_ui()
        
        # Set up drag and drop if possible
        try:
            self.root.drop_target_register("*")
            self.root.dnd_bind('<<Drop>>', self.handle_drop)
        except AttributeError:
            # No drag and drop support
            print("Note: Drag and drop support not available. Install tkinterdnd2 for this feature.")
        
    def setup_ui(self):
        """Set up the user interface with improved column handling"""
        self.root.title("PNG Metadata Viewer")
        self.root.geometry("800x600")
        
        # Create main layout frames
        self.info_frame = ttk.Frame(self.root, padding="10")
        self.info_frame.pack(fill="x")
        
        # File info labels
        self.file_label = ttk.Label(self.info_frame, text="No file loaded")
        self.file_label.pack(anchor="w")
        
        self.size_label = ttk.Label(self.info_frame, text="")
        self.size_label.pack(anchor="w")
        
        # Add search frame
        self.search_frame = ttk.Frame(self.root, padding="10")
        self.search_frame.pack(fill="x")
        
        ttk.Label(self.search_frame, text="Search:").pack(side="left", padx=(0, 5))
        
        # Search entry with validation
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self.on_search_change)
        self.search_entry = ttk.Entry(self.search_frame, textvariable=self.search_var, width=40)
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        # Search results label
        self.search_results_label = ttk.Label(self.search_frame, text="")
        self.search_results_label.pack(side="left", padx=5)
        
        # Previous/Next buttons for search
        self.prev_button = ttk.Button(self.search_frame, text="↑", width=3, 
                                      command=lambda: self.navigate_search(-1))
        self.prev_button.pack(side="left", padx=2)
        
        self.next_button = ttk.Button(self.search_frame, text="↓", width=3, 
                                     command=lambda: self.navigate_search(1))
        self.next_button.pack(side="left", padx=2)
        
        # Add separator
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=10, pady=5)
        
        # Create frame for metadata tree
        self.tree_frame = ttk.Frame(self.root, padding="10")
        self.tree_frame.pack(fill="both", expand=True)
        
        # Create treeview with scrollbars - IMPORTANT: Use two distinct columns
        self.tree = ttk.Treeview(self.tree_frame, columns=("Value",), show="tree headings")
        self.tree.heading("#0", text="Key")
        self.tree.heading("Value", text="Value")
        
        # Critical fix: Ensure columns have sufficient width and stretch properly
        self.tree.column("#0", width=300, minwidth=150, stretch=tk.YES)
        self.tree.column("Value", width=450, minwidth=200, stretch=tk.YES)

        # Add vertical scrollbar
        vsb = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        
        # Add horizontal scrollbar
        hsb = ttk.Scrollbar(self.tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand=hsb.set)
        
        # Grid layout for scrollable tree
        self.tree.grid(column=0, row=0, sticky="nsew")
        vsb.grid(column=1, row=0, sticky="ns")
        hsb.grid(column=0, row=1, sticky="ew")
        
        # Configure grid weights to ensure the tree expands properly
        self.tree_frame.columnconfigure(0, weight=1)
        self.tree_frame.rowconfigure(0, weight=1)
        
        # Add button frame
        self.button_frame = ttk.Frame(self.root, padding="10")
        self.button_frame.pack(fill="x")
        
        # Open button
        self.open_button = ttk.Button(self.button_frame, text="Open PNG File", command=self.open_file)
        self.open_button.pack(side="left", padx=5)
        
        # Copy button
        self.copy_button = ttk.Button(self.button_frame, text="Copy to Clipboard", command=self.copy_to_clipboard)
        self.copy_button.pack(side="right", padx=5)
        
        # Status label
        self.status_label = ttk.Label(self.button_frame, text="Drag & drop a PNG file to view metadata")
        self.status_label.pack(side="left", padx=20)
        
        # Keyboard bindings for search
        self.root.bind('<Control-f>', lambda e: self.search_entry.focus_set())
        self.root.bind('<F3>', lambda e: self.navigate_search(1))
        self.root.bind('<Shift-F3>', lambda e: self.navigate_search(-1))
        self.search_entry.bind('<Return>', lambda e: self.navigate_search(1))
        self.search_entry.bind('<Escape>', lambda e: self.clear_search())
        
    def clear_search(self):
        """Clear search and reset focus"""
        self.search_var.set("")
        self.search_results_label.config(text="")
        self.search_matches = []
        self.current_match = -1
        self.root.focus_set()
        
        # Remove highlight from previously highlighted items
        self._clear_all_tags()
    
    def _get_all_children(self, item=""):
        """Get all children of an item recursively (compatible with all Tkinter versions)"""
        children = self.tree.get_children(item)
        result = list(children)
        
        for child in children:
            result.extend(self._get_all_children(child))
            
        return result
    
    def _clear_all_tags(self):
        """Clear tags from all tree items (compatible with all Tkinter versions)"""
        for item_id in self._get_all_children():
            self.tree.item(item_id, tags=())
    
    def on_search_change(self, *args):
        """Handle search text changes"""
        search_text = self.search_var.get().lower()
        
        if not search_text:
            self.clear_search()
            return
            
        # Perform search
        self.search_matches = []
        for path_info in self.all_paths:
            path, node_id, value, key = path_info
            
            # Search in both keys and values
            if (search_text in key.lower() or 
                (value and search_text in str(value).lower())):
                self.search_matches.append(node_id)
        
        # Update match counter
        self.current_match = -1  # Reset position
        match_count = len(self.search_matches)
        
        if match_count > 0:
            self.search_results_label.config(text=f"0 / {match_count}")
            self.navigate_search(1)  # Go to first match
        else:
            self.search_results_label.config(text="No matches")
            
            # Remove highlight from previously highlighted items
            self._clear_all_tags()
    
    def navigate_search(self, direction):
        """Navigate between search results"""
        if not self.search_matches:
            return
            
        # Calculate new position with wraparound
        count = len(self.search_matches)
        self.current_match = (self.current_match + direction) % count
        
        # Update the counter label
        self.search_results_label.config(
            text=f"{self.current_match + 1} / {count}"
        )
        
        # Remove highlight from all items
        self._clear_all_tags()
        
        # Highlight the current match
        current_item = self.search_matches[self.current_match]
        self.tree.item(current_item, tags=('highlight',))
        
        # Configure tag
        self.tree.tag_configure('highlight', background='#FFFF99')
        
        # Ensure the item is visible
        self.tree.see(current_item)
    
    def collect_metadata_paths(self, parent="", path=None):
        """Collect all metadata key paths for searching"""
        if path is None:
            path = []
            
        paths = []
        
        for item_id in self.tree.get_children(parent):
            item = self.tree.item(item_id)
            item_text = item['text']
            item_value = item['values'][0] if item['values'] else ""
            
            # Current path to this node
            current_path = path + [item_text]
            
            # Store the full path, node ID, value, and key
            paths.append((
                '.'.join(current_path),
                item_id,
                item_value,
                item_text
            ))
            
            # Recursively get paths from children
            child_paths = self.collect_metadata_paths(item_id, current_path)
            paths.extend(child_paths)
            
        return paths

    def format_display_value(self, value):
        """Properly format and unescape potential JSON string values"""
        if isinstance(value, str):
            # Check if it's a JSON string representation of an object or array
            if (value.startswith('{') and value.endswith('}')) or \
               (value.startswith('[') and value.endswith(']')):
                try:
                    parsed = json.loads(value)
                    return parsed
                except json.JSONDecodeError:
                    pass
                    
            # Check if it's an escaped JSON string
            if "\\\"" in value or "\\n" in value or "\\[" in value or "\\]" in value:
                try:
                    # First try with double quotes
                    unescaped = json.loads(f'"{value}"')
                    return unescaped
                except json.JSONDecodeError:
                    try:
                        # Then try without added quotes
                        unescaped = json.loads(value)
                        return unescaped
                    except json.JSONDecodeError:
                        pass
                        
        return value

    def display_metadata(self, metadata: dict, file_path: str) -> None:
        """Display metadata in the tree view with character substitution to prevent truncation"""
        # Update file info
        file_name = Path(file_path).name
        file_size = Path(file_path).stat().st_size / 1024  # KB
        
        self.file_label.config(text=f"File: {file_name}")
        self.size_label.config(text=f"Size: {file_size:.1f} KB")
        
        # Store metadata for later use
        self.metadata = metadata
        
        # Clear existing tree items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Display metadata or "No metadata" message
        if not metadata:
            self.tree.insert("", "end", text="No metadata found", values=(""))
            self.status_label.config(text="No metadata found in this PNG file")
            return

        # Function to parse JSON strings
        def parse_json_string(value):
            """Parse a potential JSON string"""
            if not isinstance(value, str):
                return value
                
            # First try direct parsing
            if (value.startswith('{') and value.endswith('}')) or (value.startswith('[') and value.endswith(']')):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            
            # Try handling escaped JSON strings
            if '\\\"' in value or '\\{' in value or '\\[' in value:
                try:
                    unescaped = json.loads(f'"{value}"')
                    if isinstance(unescaped, str) and ((unescaped.startswith('{') and unescaped.endswith('}')) or 
                                                      (unescaped.startswith('[') and unescaped.endswith(']'))):
                        try:
                            return json.loads(unescaped)
                        except:
                            return unescaped
                    return unescaped
                except:
                    pass
                    
            return value
        
        # CRITICAL FIX: Function to replace spaces with non-breaking space character
        # This prevents tkinter from truncating strings at space characters
        def format_for_display(value):
            """Format a value for display, replacing spaces with non-breaking spaces"""
            if value is None:
                return ""
                
            # Convert to string if not already
            str_value = str(value)
            
            # Replace normal spaces with non-breaking spaces to prevent truncation
            # The Unicode non-breaking space character doesn't trigger the truncation
            return str_value.replace(" ", "\u00A0")
        
        # Function to add items to the tree with space substitution
        def add_item(parent, key, value):
            # Parse JSON strings first
            parsed_value = parse_json_string(value)
            
            if isinstance(parsed_value, dict):
                # Create a node for dictionary
                node = self.tree.insert(parent, "end", text=key, values=(""))
                # Add dictionary items
                for k, v in sorted(parsed_value.items()):
                    add_item(node, k, v)
            elif isinstance(parsed_value, list):
                # Format array indicator with non-breaking spaces
                array_indicator = format_for_display(f"[{len(parsed_value)} items]")
                node = self.tree.insert(parent, "end", text=key, values=(array_indicator,))
                
                # Add array elements as child nodes
                for i, element in enumerate(parsed_value):
                    # For simple types, display directly with space substitution
                    if isinstance(element, (str, int, float, bool)):
                        display_value = format_for_display(element)
                        self.tree.insert(node, "end", text=f"[{i}]", values=(display_value,))
                    else:
                        # For complex types, recurse
                        add_item(node, f"[{i}]", element)
            else:
                # For simple values, apply space substitution
                display_value = format_for_display(parsed_value)
                self.tree.insert(parent, "end", text=key, values=(display_value,))
        
        # Add metadata to tree
        for key, value in sorted(metadata.items()):
            add_item("", key, value)
        
        # Expand all nodes by default
        for item in self.tree.get_children():
            self.tree.item(item, open=True)
            
        # Collect all paths for searching
        self.all_paths = self.collect_metadata_paths()
            
        # Update status
        self.status_label.config(text=f"Found {len(metadata)} metadata entries")
    
    def load_file(self, file_path: str) -> None:
        """Load and display metadata from a file"""
        try:
            # Check if file exists and is a PNG
            path = Path(file_path)
            if not path.exists():
                messagebox.showerror("Error", f"File not found: {file_path}")
                return
                
            if path.suffix.lower() != '.png':
                messagebox.showerror("Error", f"Not a PNG file: {file_path}")
                return
                
            # Read metadata using our library
            metadata = pngmeta.read(file_path)
            
            # Display metadata
            self.current_file = file_path
            self.display_metadata(metadata, file_path)
            
            # Update window title
            self.root.title(f"PNG Metadata: {path.name}")
            
            # Clear any existing search
            self.clear_search()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def open_file(self) -> None:
        """Open file dialog to select a PNG file"""
        file_path = filedialog.askopenfilename(
            title="Select PNG File",
            filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.load_file(file_path)
    
    def handle_drop(self, event) -> None:
        """Handle drag and drop events"""
        file_path = event.data
        
        # Clean up path - sometimes drop gives us {filename}
        if file_path.startswith('{') and file_path.endswith('}'):
            file_path = file_path[1:-1]
            
        self.load_file(file_path)
    
    def copy_to_clipboard(self) -> None:
      """Copy metadata to clipboard as JSON, preserving the original format"""
      if not self.current_file:
          messagebox.showinfo("Info", "No file loaded")
          return
          
      try:
          metadata = pngmeta.read(self.current_file)
          if not metadata:
              messagebox.showinfo("Info", "No metadata to copy")
              return
              
          self.root.clipboard_clear()
          metadata_str = json.dumps(metadata, indent=2)
          self.root.clipboard_append(metadata_str)
          
          messagebox.showinfo("Copied", "Metadata copied to clipboard as JSON")
      except Exception as e:
          messagebox.showerror("Error", f"Failed to copy: {str(e)}")

def main():
    # Try to import TkinterDnD for drag and drop
    use_dnd = False
    try:
        import tkinterdnd2
        root = tkinterdnd2.TkinterDnD.Tk()
        use_dnd = True
    except ImportError:
        # Fall back to regular tkinter if TkinterDnD not available
        root = tk.Tk()
        print("Note: tkinterdnd2 module not found. Drag and drop disabled.")
        print("To enable drag and drop, install with: pip install tkinterdnd2")
    
    # Create viewer
    viewer = PNGMetadataViewer(root)
    
    # Check if a file was provided as argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        viewer.load_file(file_path)
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()