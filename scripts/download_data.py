# ABOUTME: Data download helper script for PAMCL
# ABOUTME: Downloads and prepares ASVSpoof 2019 LA dataset for training

"""
Data Download Instructions for PAMCL

The ASVSpoof 2019 LA dataset requires registration to download.
Follow these steps:

1. Register at https://www.asvspoof.org/ to get access
2. Download the following files:
   - ASVspoof2019_LA_train.zip
   - ASVspoof2019_LA_dev.zip  
   - ASVspoof2019_LA_eval.zip
   - ASVspoof2019_LA_cm_protocols.zip

3. Extract all files to: pamcl/data/ASVspoof2019_LA/

Expected directory structure:
pamcl/data/ASVspoof2019_LA/
├── ASVspoof2019_LA_train/
│   └── flac/
│       └── *.flac files
├── ASVspoof2019_LA_dev/
│   └── flac/
│       └── *.flac files
├── ASVspoof2019_LA_eval/
│   └── flac/
│       └── *.flac files
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt

4. Run this script to verify the data structure:
   python scripts/download_data.py --verify
"""

import argparse
import os
from pathlib import Path


def verify_dataset(base_path: Path) -> bool:
    """Verify that ASVSpoof 2019 LA dataset is properly structured.
    
    Args:
        base_path: Path to ASVspoof2019_LA directory
        
    Returns:
        True if verification passed
    """
    required_dirs = [
        'ASVspoof2019_LA_train/flac',
        'ASVspoof2019_LA_dev/flac',
        'ASVspoof2019_LA_eval/flac',
        'ASVspoof2019_LA_cm_protocols'
    ]
    
    required_protocols = [
        'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
        'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
        'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
    ]
    
    print(f"Verifying dataset at: {base_path}")
    all_valid = True
    
    # Check directories
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            if 'flac' in dir_path:
                flac_count = len(list(full_path.glob('*.flac')))
                print(f"  [OK] {dir_path} ({flac_count} files)")
            else:
                print(f"  [OK] {dir_path}")
        else:
            print(f"  [MISSING] {dir_path}")
            all_valid = False
    
    # Check protocol files
    for proto_path in required_protocols:
        full_path = base_path / proto_path
        if full_path.exists():
            with open(full_path, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"  [OK] {proto_path} ({line_count} entries)")
        else:
            print(f"  [MISSING] {proto_path}")
            all_valid = False
    
    if all_valid:
        print("\nDataset verification PASSED!")
        print("You can now run: python train.py --config configs/default.yaml")
    else:
        print("\nDataset verification FAILED!")
        print("Please download missing files from https://www.asvspoof.org/")
    
    return all_valid


def print_download_instructions():
    """Print download instructions."""
    print(__doc__)


def main():
    parser = argparse.ArgumentParser(description='ASVSpoof 2019 LA data preparation')
    parser.add_argument(
        '--verify', action='store_true',
        help='Verify existing dataset structure'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/ASVspoof2019_LA',
        help='Path to dataset directory'
    )
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    data_path = script_dir / args.data_dir
    
    if args.verify:
        verify_dataset(data_path)
    else:
        print_download_instructions()


if __name__ == '__main__':
    main()
