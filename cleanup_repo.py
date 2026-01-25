import os
import shutil
from pathlib import Path

def cleanup_repo():
    """
    Clean up the repository by organizing files into a clear structure:
    - Keep essential files and final outputs
    - Move experimental/deprecated scripts to archive/
    - Maintain a clean, readable structure for submission
    """
    
    # Define the files/directories to keep
    essential_files = [
        'README.md',
        'requirements.txt',
        'utils.py',
        'sanity_check_inference.py',
        'sanity_check_comparison.png',
        'data/',
        'outputs/'
    ]
    
    # Define canonical scripts to keep
    canonical_scripts = [
        'run_clipseg.py',  # Phase 2: zero-shot baseline
        'run_ensemble_clipseg.py',  # Phase 3A: ensembling
        'run_phase3b_rtx4090.py'  # Phase 3B: fine-tuning
    ]
    
    # Create archive directory if it doesn't exist
    archive_dir = Path('archive')
    archive_dir.mkdir(exist_ok=True)
    
    # Create scripts directory if it doesn't exist
    scripts_dir = Path('scripts')
    scripts_dir.mkdir(exist_ok=True)
    
    # Move canonical scripts to scripts/ directory with proper names
    canonical_mapping = {
        'run_clipseg.py': 'phase2_baseline.py',
        'run_ensemble_clipseg.py': 'phase3a_ensemble.py',
        'run_phase3b_rtx4090.py': 'phase3b_finetune.py'
    }
    
    for old_name, new_name in canonical_mapping.items():
        if os.path.exists(old_name):
            dest_path = scripts_dir / new_name
            if not dest_path.exists():  # Only move if destination doesn't exist
                shutil.move(old_name, dest_path)
                print(f"Moved {old_name} to scripts/{new_name}")
    
    # Identify files to archive (non-canonical scripts)
    files_to_archive = []
    for file in os.listdir('.'):
        if os.path.isfile(file) and file.endswith('.py'):
            if file not in canonical_scripts and file != 'cleanup_repo.py' and file != 'sanity_check_inference.py':
                files_to_archive.append(file)
    
    # Move non-essential scripts to archive
    for file in files_to_archive:
        if os.path.exists(file):
            dest_path = archive_dir / file
            if not dest_path.exists():  # Only move if destination doesn't exist
                shutil.move(file, dest_path)
                print(f"Archived {file}")
    
    # Create data/datasets directory if it doesn't exist
    datasets_dir = Path('data/datasets')
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # Print summary
    print("\nCleanup completed. Summary:")
    print("- Essential files preserved")
    print("- Canonical scripts moved to scripts/ directory")
    print("- Experimental/deprecated scripts moved to archive/")
    print("- Directory structure organized for submission")
    print("- Outputs and results preserved")

if __name__ == "__main__":
    cleanup_repo()