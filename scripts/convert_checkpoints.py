#!/usr/bin/env python3
"""
Standalone script to convert original D3 checkpoints to Lightning format.
Usage: python convert_checkpoints.py <input_dir> [output_dir]
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.checkpoint_utils import convert_pth_to_ckpt, is_original_checkpoint
from utils.utils import load_hydra_config_from_run


def main():
    parser = argparse.ArgumentParser(description='Convert D3 checkpoints to Lightning format')
    parser.add_argument('input_dir', help='Directory containing original checkpoints')
    parser.add_argument('--output_dir', default=None, help='Output directory (default: same as input)')
    parser.add_argument('--recursive', '-r', action='store_true', help='Search recursively')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return 1
    
    # Find all .pth files
    if args.recursive:
        pth_files = list(input_dir.rglob('*.pth'))
    else:
        pth_files = list(input_dir.glob('*.pth'))
    
    print(f"Found {len(pth_files)} .pth files")
    
    converted = 0
    for pth_file in pth_files:
        if is_original_checkpoint(str(pth_file)):
            # Create corresponding .ckpt path
            rel_path = pth_file.relative_to(input_dir)
            ckpt_path = output_dir / rel_path.with_suffix('.ckpt')
            
            try:
                # Try to load config from the same directory
                cfg = None
                config_path = pth_file.parent / 'hydra' / 'config.yaml'
                if config_path.exists():
                    try:
                        cfg = load_hydra_config_from_run(str(pth_file.parent))
                    except:
                        print(f"Warning: Could not load config for {pth_file}")
                
                convert_pth_to_ckpt(str(pth_file), str(ckpt_path), cfg)
                converted += 1
                
            except Exception as e:
                print(f"Error converting {pth_file}: {e}")
        else:
            print(f"Skipping {pth_file} (not original D3 format)")
    
    print(f"Successfully converted {converted} checkpoints")
    return 0


if __name__ == '__main__':
    sys.exit(main())