#!/usr/bin/env python3
"""
Migration script to replace print statements with centralized logging.

This script replaces all print() calls with proper logging throughout the codebase.
"""

import re
import os
from pathlib import Path
from src.logging_config import get_logger

logger = get_logger(__name__)


def add_logging_import(content: str, module_path: str) -> str:
    """Add logging import to a Python file."""
    # Determine the correct import path
    if module_path.startswith('src/'):
        import_line = "from .logging_config import get_logger"
    else:
        import_line = "from src.logging_config import get_logger"
    
    logger_init = f"\nlogger = get_logger(__name__)"
    
    # Find the last import line
    lines = content.split('\n')
    last_import_idx = -1
    
    for i, line in enumerate(lines):
        if (line.startswith('import ') or line.startswith('from ')) and not line.strip().startswith('#'):
            last_import_idx = i
    
    if last_import_idx >= 0:
        # Insert after the last import
        lines.insert(last_import_idx + 1, import_line)
        lines.insert(last_import_idx + 2, logger_init)
    else:
        # No imports found, add at the beginning after docstring
        doc_end = 0
        if lines and lines[0].strip().startswith('"""'):
            # Find end of docstring
            for i in range(1, len(lines)):
                if '"""' in lines[i]:
                    doc_end = i + 1
                    break
        lines.insert(doc_end, import_line)
        lines.insert(doc_end + 1, logger_init)
    
    return '\n'.join(lines)


def replace_print_statements(content: str) -> str:
    """Replace print statements with appropriate logging calls."""
    
    # Pattern to match print statements
    print_pattern = r'print\(f?"([^"]*)"[^)]*\)'
    
    def replace_print(match):
        message = match.group(1)
        full_match = match.group(0)
        
        # Determine log level based on content
        if any(word in message.lower() for word in ['error', 'failed', 'exception']):
            level = 'error'
        elif any(word in message.lower() for word in ['warning', 'warn']):
            level = 'warning'
        elif any(word in message.lower() for word in ['debug']):
            level = 'debug'
        else:
            level = 'info'
        
        # Extract the f-string content if it's an f-string
        if full_match.startswith('print(f"'):
            # It's an f-string
            return f'logger.{level}(f"{message}")'
        else:
            # Regular string
            return f'logger.{level}("{message}")'
    
    # Replace print statements
    result = re.sub(print_pattern, replace_print, content)
    
    # Handle more complex print statements
    complex_patterns = [
        (r'print\(f"([^"]*)"\.format\([^)]*\)\)', r'logger.info(f"\1")'),
        (r'print\("([^"]*)"\.format\([^)]*\)\)', r'logger.info("\1")'),
    ]
    
    for pattern, replacement in complex_patterns:
        result = re.sub(pattern, replacement, result)
    
    return result


def migrate_file(file_path: Path) -> bool:
    """Migrate a single Python file to use logging."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file already has logging
        if 'logger = get_logger(__name__)' in content:
            logger.info(f"Skipping {file_path} - already migrated")
            return False
        
        # Check if file has print statements
        if 'print(' not in content:
            logger.info(f"Skipping {file_path} - no print statements")
            return False
        
        logger.info(f"Migrating {file_path}...")
        
        # Add logging imports
        modified_content = add_logging_import(content, str(file_path))
        
        # Replace print statements
        modified_content = replace_print_statements(modified_content)
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        return True
        
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"File access error migrating {file_path}: {e}")
        return False
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error migrating {file_path}: {e}")
        return False
    except (OSError, IOError) as e:
        logger.error(f"I/O error migrating {file_path}: {e}")
        return False


def main():
    """Main migration function."""
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
    
    migrated_count = 0
    for file_path in python_files:
        if migrate_file(file_path):
            migrated_count += 1
    
    logger.info(f"\nMigration complete! Migrated {migrated_count} files.")


if __name__ == "__main__":
    main()