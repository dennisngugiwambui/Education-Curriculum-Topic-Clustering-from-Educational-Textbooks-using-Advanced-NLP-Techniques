#!/usr/bin/env python3
"""
Rename all files with 'masters' to professional names
"""

import os
import shutil
from pathlib import Path

def rename_masters_files():
    """Rename all files containing 'masters' to professional alternatives"""
    
    base_path = Path(r"c:\Users\Denno\Desktop\my assignment")
    
    # Find all files with 'masters' in the name
    masters_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if 'masters' in file.lower():
                masters_files.append(Path(root) / file)
    
    print(f"Found {len(masters_files)} files to rename:")
    
    for old_path in masters_files:
        # Create new name by replacing 'masters_research' with 'curriculum_analysis'
        new_name = old_path.name.replace('masters_research_', 'curriculum_analysis_')
        new_name = new_name.replace('masters_', 'curriculum_')
        new_path = old_path.parent / new_name
        
        try:
            shutil.move(str(old_path), str(new_path))
            print(f"✅ Renamed: {old_path.name} -> {new_name}")
        except Exception as e:
            print(f"❌ Failed to rename {old_path.name}: {e}")

if __name__ == "__main__":
    rename_masters_files()
    print("\n🎉 File renaming completed!")
