#!/usr/bin/env python3
"""Automated dependency update script with safety checks."""

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re


class DependencyUpdater:
    """Manages dependency updates with safety checks."""
    
    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path or Path(__file__).parent.parent
        self.backup_dir = self.repo_path / '.dependency_backups'
        self.backup_dir.mkdir(exist_ok=True)
        
    def backup_requirements(self) -> str:
        """Create backup of current requirements files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f'backup_{timestamp}'
        backup_path.mkdir(exist_ok=True)
        
        req_files = [
            'requirements.txt',
            'requirements-dev.txt', 
            'pyproject.toml'
        ]
        
        for req_file in req_files:
            src_path = self.repo_path / req_file
            if src_path.exists():
                dst_path = backup_path / req_file
                subprocess.run(['cp', str(src_path), str(dst_path)], check=True)
                
        print(f"📦 Created backup at {backup_path}")
        return str(backup_path)
    
    def get_outdated_packages(self) -> List[Dict[str, str]]:
        """Get list of outdated packages."""
        try:
            result = subprocess.run(
                ['pip', 'list', '--outdated', '--format=json'],
                capture_output=True, text=True, check=True
            )
            outdated = json.loads(result.stdout)
            return outdated
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return []
    
    def check_security_vulnerabilities(self) -> List[Dict]:
        """Check for security vulnerabilities in current dependencies."""
        try:
            result = subprocess.run(
                ['safety', 'check', '--json'],
                capture_output=True, text=True
            )
            if result.stdout.strip():
                return json.loads(result.stdout)
            return []
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            print("⚠️ Safety check failed or not available")
            return []
    
    def update_package(self, package_name: str, new_version: str = None) -> bool:
        """Update a specific package."""
        try:
            if new_version:
                cmd = ['pip', 'install', f'{package_name}=={new_version}']
            else:
                cmd = ['pip', 'install', '--upgrade', package_name]
                
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Updated {package_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to update {package_name}: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run tests to verify updates don't break functionality."""
        try:
            print("🧪 Running tests to verify updates...")
            result = subprocess.run(
                ['python', '-m', 'pytest', 'tests/', '-x', '--tb=short'],
                cwd=self.repo_path, capture_output=True, text=True, timeout=300
            )
            
            if result.returncode == 0:
                print("✅ All tests passed")
                return True
            else:
                print(f"❌ Tests failed:\n{result.stdout}\n{result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("⏰ Tests timed out")
            return False
        except subprocess.CalledProcessError as e:
            print(f"❌ Test execution failed: {e}")
            return False
    
    def check_import_compatibility(self) -> bool:
        """Check if all imports still work after updates."""
        try:
            print("🔍 Checking import compatibility...")
            
            # Try to import main modules
            test_imports = [
                'import src.api',
                'import src.train_model',
                'import src.predict_churn',
                'import src.preprocess_data'
            ]
            
            for import_stmt in test_imports:
                result = subprocess.run(
                    ['python', '-c', import_stmt],
                    cwd=self.repo_path, capture_output=True, text=True
                )
                if result.returncode != 0:
                    print(f"❌ Import failed: {import_stmt}")
                    print(f"Error: {result.stderr}")
                    return False
                    
            print("✅ All imports successful")
            return True
        except Exception as e:
            print(f"❌ Import check failed: {e}")
            return False
    
    def generate_lock_file(self) -> bool:
        """Generate lock file with exact versions."""
        try:
            print("🔒 Generating lock file...")
            
            # Generate requirements lock
            result = subprocess.run(
                ['pip', 'freeze'],
                capture_output=True, text=True, check=True
            )
            
            lock_file = self.repo_path / 'requirements.lock'
            with open(lock_file, 'w') as f:
                f.write(f"# Generated on {datetime.now().isoformat()}\n")
                f.write(f"# Python version: {sys.version.split()[0]}\n\n")
                f.write(result.stdout)
                
            print(f"✅ Lock file generated: {lock_file}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to generate lock file: {e}")
            return False
    
    def restore_backup(self, backup_path: str) -> bool:
        """Restore from backup if updates fail."""
        try:
            print(f"🔄 Restoring from backup: {backup_path}")
            
            backup_dir = Path(backup_path)
            req_files = ['requirements.txt', 'requirements-dev.txt', 'pyproject.toml']
            
            for req_file in req_files:
                backup_file = backup_dir / req_file
                target_file = self.repo_path / req_file
                
                if backup_file.exists():
                    subprocess.run(['cp', str(backup_file), str(target_file)], check=True)
                    
            # Reinstall from backup
            subprocess.run(
                ['pip', 'install', '-r', str(self.repo_path / 'requirements.txt')],
                check=True
            )
            
            print("✅ Successfully restored from backup")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to restore backup: {e}")
            return False
    
    def update_requirements_file(self, packages_updated: List[str]) -> bool:
        """Update requirements.txt with new versions."""
        try:
            req_file = self.repo_path / 'requirements.txt'
            if not req_file.exists():
                return True
                
            # Get current installed versions
            result = subprocess.run(
                ['pip', 'freeze'],
                capture_output=True, text=True, check=True
            )
            
            installed_versions = {}
            for line in result.stdout.split('\n'):
                if '==' in line:
                    name, version = line.split('==', 1)
                    installed_versions[name.lower()] = version
                    
            # Update requirements file
            with open(req_file, 'r') as f:
                lines = f.readlines()
                
            updated_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name
                    package_name = re.split('[>=<~!]', line)[0].strip()
                    if package_name.lower() in [p.lower() for p in packages_updated]:
                        if package_name.lower() in installed_versions:
                            new_line = f"{package_name}=={installed_versions[package_name.lower()]}"
                            updated_lines.append(new_line + '\n')
                            print(f"📝 Updated {package_name} in requirements.txt")
                        else:
                            updated_lines.append(line + '\n')
                    else:
                        updated_lines.append(line + '\n')
                else:
                    updated_lines.append(line + '\n' if not line.endswith('\n') else line)
                    
            with open(req_file, 'w') as f:
                f.writelines(updated_lines)
                
            return True
        except Exception as e:
            print(f"❌ Failed to update requirements file: {e}")
            return False
    
    def create_update_summary(self, updated_packages: List[Dict], 
                            security_fixes: List[Dict]) -> str:
        """Create summary of updates performed."""
        summary = []
        summary.append("# Dependency Update Summary")
        summary.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        if updated_packages:
            summary.append("## Updated Packages")
            for pkg in updated_packages:
                summary.append(f"- **{pkg['name']}**: {pkg.get('old_version', 'unknown')} → {pkg.get('new_version', 'latest')}")
            summary.append("")
            
        if security_fixes:
            summary.append("## Security Fixes")
            for fix in security_fixes:
                summary.append(f"- **{fix.get('package', 'unknown')}**: Fixed vulnerability {fix.get('id', 'unknown')}")
            summary.append("")
            
        summary.append("## Verification")
        summary.append("- ✅ Tests passed")
        summary.append("- ✅ Import compatibility verified")
        summary.append("- ✅ Lock file generated")
        
        return '\n'.join(summary)
    
    def safe_update(self, packages: List[str] = None, 
                   security_only: bool = False,
                   max_updates: int = 10) -> bool:
        """Perform safe dependency updates with rollback capability."""
        print("🔄 Starting safe dependency update process...")
        
        # Create backup
        backup_path = self.backup_requirements()
        
        try:
            # Get outdated packages
            outdated = self.get_outdated_packages()
            if not outdated:
                print("✅ All packages are up to date")
                return True
                
            print(f"📦 Found {len(outdated)} outdated packages")
            
            # Get security vulnerabilities
            vulnerabilities = self.check_security_vulnerabilities()
            vulnerable_packages = [v.get('package', '') for v in vulnerabilities]
            
            # Determine packages to update
            packages_to_update = []
            
            if security_only:
                # Only update packages with security vulnerabilities
                packages_to_update = [
                    pkg for pkg in outdated 
                    if pkg['name'] in vulnerable_packages
                ]
                print(f"🔒 Security-only mode: updating {len(packages_to_update)} vulnerable packages")
            elif packages:
                # Update specific packages
                packages_to_update = [
                    pkg for pkg in outdated 
                    if pkg['name'] in packages
                ]
                print(f"🎯 Targeted update: updating {len(packages_to_update)} specified packages")
            else:
                # Update all outdated packages (limited by max_updates)
                packages_to_update = outdated[:max_updates]
                print(f"🚀 Updating {len(packages_to_update)} packages (limited to {max_updates})")
            
            if not packages_to_update:
                print("ℹ️ No packages to update")
                return True
                
            updated_packages = []
            failed_updates = []
            
            # Update packages one by one
            for pkg in packages_to_update:
                print(f"\n📦 Updating {pkg['name']} from {pkg['version']} to {pkg['latest_version']}")
                
                if self.update_package(pkg['name']):
                    # Verify the update doesn't break imports
                    if self.check_import_compatibility():
                        updated_packages.append({
                            'name': pkg['name'],
                            'old_version': pkg['version'],
                            'new_version': pkg['latest_version']
                        })
                        print(f"✅ Successfully updated {pkg['name']}")
                    else:
                        print(f"❌ Update of {pkg['name']} broke compatibility, rolling back...")
                        self.update_package(pkg['name'], pkg['version'])
                        failed_updates.append(pkg['name'])
                else:
                    failed_updates.append(pkg['name'])
            
            if not updated_packages:
                print("ℹ️ No packages were successfully updated")
                return True
                
            # Run comprehensive tests
            if self.run_tests():
                print("✅ All tests passed after updates")
                
                # Update requirements file
                package_names = [pkg['name'] for pkg in updated_packages]
                if self.update_requirements_file(package_names):
                    print("✅ Updated requirements.txt")
                    
                # Generate lock file
                if self.generate_lock_file():
                    print("✅ Generated lock file")
                    
                # Create summary
                security_fixes = [v for v in vulnerabilities if v.get('package') in package_names]
                summary = self.create_update_summary(updated_packages, security_fixes)
                
                summary_file = self.repo_path / 'dependency_update_summary.md'
                with open(summary_file, 'w') as f:
                    f.write(summary)
                    
                print(f"📊 Update summary saved to {summary_file}")
                print(f"✅ Successfully updated {len(updated_packages)} packages")
                
                if failed_updates:
                    print(f"⚠️ Failed to update: {', '.join(failed_updates)}")
                    
                return True
            else:
                print("❌ Tests failed after updates, rolling back...")
                self.restore_backup(backup_path)
                return False
                
        except Exception as e:
            print(f"❌ Update process failed: {e}")
            print("🔄 Rolling back changes...")
            self.restore_backup(backup_path)
            return False


def main():
    """Main function for dependency updates."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Safe dependency updater')
    parser.add_argument('--packages', nargs='+', help='Specific packages to update')
    parser.add_argument('--security-only', action='store_true', 
                       help='Only update packages with security vulnerabilities')
    parser.add_argument('--max-updates', type=int, default=10,
                       help='Maximum number of packages to update at once')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be updated without making changes')
    
    args = parser.parse_args()
    
    updater = DependencyUpdater()
    
    if args.dry_run:
        print("🔍 Dry run mode - showing what would be updated:")
        outdated = updater.get_outdated_packages()
        vulnerabilities = updater.check_security_vulnerabilities()
        
        if args.security_only:
            vulnerable_packages = [v.get('package', '') for v in vulnerabilities]
            to_update = [pkg for pkg in outdated if pkg['name'] in vulnerable_packages]
        elif args.packages:
            to_update = [pkg for pkg in outdated if pkg['name'] in args.packages]
        else:
            to_update = outdated[:args.max_updates]
            
        if to_update:
            print(f"Would update {len(to_update)} packages:")
            for pkg in to_update:
                print(f"  - {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
        else:
            print("No packages would be updated")
            
        return 0
    
    success = updater.safe_update(
        packages=args.packages,
        security_only=args.security_only,
        max_updates=args.max_updates
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())