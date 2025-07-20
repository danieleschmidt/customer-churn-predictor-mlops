#!/usr/bin/env python3
"""
Verification script to ensure logging migration was successful.
"""

import os
import re
from pathlib import Path


def check_file_for_logging(file_path: Path) -> dict:
    """Check a Python file for proper logging usage."""
    result = {
        'file': str(file_path),
        'has_logger_import': False,
        'has_logger_init': False,
        'print_statements': [],
        'logging_calls': 0,
        'status': 'unknown'
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for logging imports
        if 'from .logging_config import get_logger' in content or 'from src.logging_config import get_logger' in content:
            result['has_logger_import'] = True
        
        # Check for logger initialization
        if 'logger = get_logger(__name__)' in content:
            result['has_logger_init'] = True
        
        # Find remaining print statements
        print_pattern = r'print\('
        print_matches = re.findall(print_pattern, content)
        result['print_statements'] = print_matches
        
        # Count logging calls
        logging_pattern = r'logger\.(debug|info|warning|error|critical)\('
        logging_matches = re.findall(logging_pattern, content)
        result['logging_calls'] = len(logging_matches)
        
        # Determine status
        if not result['has_logger_import'] and not result['has_logger_init'] and not print_matches:
            result['status'] = 'no_logging_needed'
        elif result['has_logger_import'] and result['has_logger_init'] and not print_matches:
            result['status'] = 'fully_migrated'
        elif result['has_logger_import'] and result['has_logger_init'] and print_matches:
            result['status'] = 'partially_migrated'
        elif print_matches:
            result['status'] = 'needs_migration'
        else:
            result['status'] = 'check_manually'
            
    except Exception as e:
        result['status'] = f'error: {e}'
    
    return result


def main():
    """Main verification function."""
    root_dir = Path(__file__).parent
    
    # Find all Python files
    python_files = []
    for pattern in ['src/**/*.py', 'scripts/**/*.py']:
        python_files.extend(root_dir.glob(pattern))
    
    # Exclude test files and __pycache__
    python_files = [
        f for f in python_files 
        if not any(part.startswith('__pycache__') or part.startswith('test_') for part in f.parts)
    ]
    
    results = []
    for file_path in python_files:
        result = check_file_for_logging(file_path)
        results.append(result)
    
    # Summary
    status_counts = {}
    for result in results:
        status = result['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("Logging Migration Verification Report")
    print("=" * 40)
    
    for status, count in status_counts.items():
        print(f"{status}: {count} files")
    
    print("\nDetailed Results:")
    print("-" * 40)
    
    for result in results:
        file_name = result['file'].split('/')[-1]
        status = result['status']
        logging_calls = result['logging_calls']
        print_count = len(result['print_statements'])
        
        print(f"{file_name:25} | {status:20} | {logging_calls:3} logs | {print_count:3} prints")
    
    # Show files that need attention
    needs_attention = [r for r in results if r['status'] in ['needs_migration', 'partially_migrated']]
    
    if needs_attention:
        print(f"\nFiles needing attention ({len(needs_attention)}):")
        print("-" * 40)
        for result in needs_attention:
            print(f"• {result['file']}: {result['status']}")
            if result['print_statements']:
                print(f"  - {len(result['print_statements'])} print statements remaining")
    
    total_migrated = status_counts.get('fully_migrated', 0)
    total_files = len([r for r in results if r['status'] != 'no_logging_needed'])
    
    if total_files > 0:
        migration_percentage = (total_migrated / total_files) * 100
        print(f"\nMigration Progress: {total_migrated}/{total_files} files ({migration_percentage:.1f}%)")
    
    return needs_attention


if __name__ == "__main__":
    main()