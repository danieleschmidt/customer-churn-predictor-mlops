#!/usr/bin/env python3
"""
Dependency validation script for reproducible builds.

This script validates that:
1. All dependencies in requirements files have exact versions
2. No dependency conflicts exist
3. Critical security packages are up to date
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple


def parse_requirements_file(file_path: Path) -> Dict[str, str]:
    """Parse a requirements file and return package name -> version mapping."""
    requirements = {}
    
    if not file_path.exists():
        return requirements
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Skip -r includes for now
            if line.startswith('-r'):
                continue
            
            # Parse package==version
            if '==' in line:
                package, version = line.split('==', 1)
                requirements[package.strip()] = version.strip()
            elif '>=' in line:
                # Minimum version constraint
                package, version = line.split('>=', 1)
                requirements[package.strip()] = f">={version.strip()}"
            else:
                print(f"⚠️  Warning: Unpinned dependency at {file_path}:{line_num}: {line}")
    
    return requirements


def validate_exact_versions(requirements: Dict[str, str], file_name: str) -> List[str]:
    """Validate that all dependencies have exact versions."""
    issues = []
    
    for package, version in requirements.items():
        if not version or version.startswith('>=') or version.startswith('>') or version.startswith('~'):
            issues.append(f"❌ {file_name}: {package} does not have exact version (found: {version})")
    
    return issues


def check_critical_packages(requirements: Dict[str, str]) -> List[str]:
    """Check if critical security packages are present and reasonably up-to-date."""
    issues = []
    
    critical_packages = {
        'requests': {'min_major': 2, 'min_minor': 28},  # Security fixes
        'urllib3': {'min_major': 1, 'min_minor': 26},   # Security fixes
        'certifi': {'min_year': 2023},                  # SSL certificates
    }
    
    for package, constraints in critical_packages.items():
        if package not in requirements:
            continue
            
        version = requirements[package]
        if version.startswith('>='):
            continue  # Skip constraint-only versions
            
        try:
            # Parse version (simplified - only handles X.Y.Z format)
            version_parts = version.split('.')
            major = int(version_parts[0])
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0
            
            if 'min_major' in constraints and major < constraints['min_major']:
                issues.append(f"🔒 Security: {package} v{version} is below recommended minimum")
            elif 'min_minor' in constraints and major == constraints['min_major'] and minor < constraints['min_minor']:
                issues.append(f"🔒 Security: {package} v{version} is below recommended minimum")
                
        except (ValueError, IndexError):
            issues.append(f"⚠️  Warning: Could not parse version for {package}: {version}")
    
    return issues


def find_duplicate_packages(prod_reqs: Dict[str, str], dev_reqs: Dict[str, str]) -> List[str]:
    """Find packages that appear in both production and development requirements."""
    issues = []
    
    overlapping = set(prod_reqs.keys()) & set(dev_reqs.keys())
    
    for package in overlapping:
        prod_version = prod_reqs[package]
        dev_version = dev_reqs[package]
        
        if prod_version != dev_version:
            issues.append(f"🔄 Version mismatch: {package} (prod: {prod_version}, dev: {dev_version})")
    
    return issues


def main():
    """Main validation function."""
    root_dir = Path(__file__).parent
    
    print("🔍 Dependency Validation Report")
    print("=" * 50)
    
    # Parse requirements files
    prod_reqs = parse_requirements_file(root_dir / "requirements.txt")
    dev_reqs = parse_requirements_file(root_dir / "requirements-dev.txt")
    prod_lock = parse_requirements_file(root_dir / "requirements.lock")
    dev_lock = parse_requirements_file(root_dir / "requirements-dev.lock")
    
    all_issues = []
    
    # Validate exact versions in base requirements
    all_issues.extend(validate_exact_versions(prod_reqs, "requirements.txt"))
    all_issues.extend(validate_exact_versions(dev_reqs, "requirements-dev.txt"))
    
    # Check for security-critical packages
    all_issues.extend(check_critical_packages(prod_reqs))
    all_issues.extend(check_critical_packages(prod_lock))
    
    # Check for version mismatches
    all_issues.extend(find_duplicate_packages(prod_reqs, dev_reqs))
    
    # Validate lockfiles exist and have content
    if not prod_lock:
        all_issues.append("❌ Missing or empty requirements.lock file")
    
    if not dev_lock:
        all_issues.append("❌ Missing or empty requirements-dev.lock file")
    
    # Report results
    if all_issues:
        print("\n📋 Issues Found:")
        for issue in all_issues:
            print(f"  {issue}")
    else:
        print("\n✅ All dependency validations passed!")
    
    # Summary statistics
    print(f"\n📊 Summary:")
    print(f"  Production packages: {len(prod_reqs)}")
    print(f"  Development packages: {len(dev_reqs)}")
    print(f"  Production lockfile entries: {len(prod_lock)}")
    print(f"  Development lockfile entries: {len(dev_lock)}")
    print(f"  Issues found: {len(all_issues)}")
    
    # Return exit code
    return len(all_issues)


if __name__ == "__main__":
    sys.exit(main())