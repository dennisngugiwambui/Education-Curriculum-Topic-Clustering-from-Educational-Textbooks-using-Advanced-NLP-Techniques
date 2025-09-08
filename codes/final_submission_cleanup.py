#!/usr/bin/env python3
"""
Final Submission Directory Cleanup
Removes temporary and duplicate files for professional submission
"""

import os
import shutil
from pathlib import Path

def cleanup_directory():
    """Clean up the codes directory for final submission"""
    
    # Files to remove (temporary/duplicate/test files)
    files_to_remove = [
        'display_results.py',
        'jupyter_results_fixed.py', 
        'notebook_results_cell.py',
        'professional_results_analysis.py',
        'results_visualization.py',
        'simple_results_viewer.py',
        'updated_cell3.py',
        'main_runner.py',  # Duplicate of main functionality
        'complete_curriculum_analysis.py'  # Integrated into main.py
    ]
    
    # Remove temporary files
    removed_count = 0
    for filename in files_to_remove:
        filepath = Path(filename)
        if filepath.exists():
            try:
                os.remove(filepath)
                print(f"✅ Removed: {filename}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Could not remove {filename}: {e}")
    
    # Rename files to remove 'masters_research_' prefix
    rename_mappings = {
        'masters_research_comprehensive_report_20250908_005917.json': 'research_comprehensive_report.json',
        'masters_research_dataset_20250908_005652.csv': 'research_dataset.csv',
        'masters_research_dataset_20250908_005652.json': 'research_dataset.json',
        'masters_research_dataset_stats_20250908_005652.json': 'research_dataset_stats.json',
        'masters_research_nlp_results_20250908_005839.csv': 'research_nlp_results.csv'
    }
    
    renamed_count = 0
    for old_name, new_name in rename_mappings.items():
        old_path = Path(old_name)
        new_path = Path(new_name)
        
        if old_path.exists() and not new_path.exists():
            try:
                shutil.move(str(old_path), str(new_path))
                print(f"✅ Renamed: {old_name} → {new_name}")
                renamed_count += 1
            except Exception as e:
                print(f"❌ Could not rename {old_name}: {e}")
    
    # Clean up __pycache__ directory
    pycache_dir = Path('__pycache__')
    if pycache_dir.exists():
        try:
            shutil.rmtree(pycache_dir)
            print(f"✅ Removed: __pycache__ directory")
        except Exception as e:
            print(f"❌ Could not remove __pycache__: {e}")
    
    print(f"\n📊 CLEANUP SUMMARY:")
    print(f"   Files removed: {removed_count}")
    print(f"   Files renamed: {renamed_count}")
    print(f"   Directory cleaned: __pycache__")
    
    # List final files
    print(f"\n📁 FINAL SUBMISSION FILES:")
    final_files = sorted([f for f in os.listdir('.') if os.path.isfile(f)])
    for i, filename in enumerate(final_files, 1):
        print(f"   {i:2d}. {filename}")
    
    print(f"\n✅ Directory prepared for professional submission!")

if __name__ == "__main__":
    print("🧹 Cleaning up codes directory for final submission...")
    cleanup_directory()
