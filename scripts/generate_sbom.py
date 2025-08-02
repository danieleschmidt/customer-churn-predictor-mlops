#!/usr/bin/env python3
"""Generate Software Bill of Materials (SBOM) for the project."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pkg_resources


def get_git_info() -> Dict[str, str]:
    """Get Git repository information."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        commit_date = subprocess.check_output(
            ["git", "log", "-1", "--format=%cI"], text=True
        ).strip()
        repo_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], text=True
        ).strip()
        
        # Clean up repo URL if it's SSH format
        if repo_url.startswith("git@"):
            repo_url = repo_url.replace("git@github.com:", "https://github.com/")
        if repo_url.endswith(".git"):
            repo_url = repo_url[:-4]
            
        return {
            "commit_hash": commit_hash,
            "commit_date": commit_date,
            "repository_url": repo_url
        }
    except subprocess.CalledProcessError:
        return {
            "commit_hash": "unknown",
            "commit_date": datetime.now().isoformat(),
            "repository_url": "unknown"
        }


def get_python_packages() -> List[Dict[str, Any]]:
    """Get list of installed Python packages."""
    packages = []
    
    try:
        # Get packages from pip freeze
        result = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        
        for line in result.strip().split('\n'):
            if line and not line.startswith('#') and '==' in line:
                name, version = line.split('==', 1)
                
                # Try to get additional package info
                try:
                    dist = pkg_resources.get_distribution(name)
                    location = dist.location
                    homepage = getattr(dist, 'homepage', '')
                    author = getattr(dist, 'author', '')
                except Exception:
                    location = ''
                    homepage = ''
                    author = ''
                
                packages.append({
                    "name": name,
                    "version": version,
                    "type": "python-package",
                    "location": location,
                    "homepage": homepage,
                    "author": author,
                    "supplier": "PyPI"
                })
                
    except subprocess.CalledProcessError:
        print("Warning: Could not get package list from pip freeze", file=sys.stderr)
    
    return packages


def get_system_packages() -> List[Dict[str, Any]]:
    """Get system-level dependencies (basic detection)."""
    packages = []
    
    # Check for common system dependencies
    system_deps = [
        "python3",
        "python3-dev", 
        "gcc",
        "g++",
        "make",
        "pkg-config"
    ]
    
    for dep in system_deps:
        try:
            # Try to get version info
            result = subprocess.check_output(
                ["dpkg", "-l", dep], 
                text=True, 
                stderr=subprocess.DEVNULL
            )
            # Parse dpkg output (simplified)
            lines = result.strip().split('\n')
            for line in lines:
                if line.startswith('ii'):
                    parts = line.split()
                    if len(parts) >= 3:
                        packages.append({
                            "name": parts[1],
                            "version": parts[2],
                            "type": "system-package",
                            "supplier": "Debian/Ubuntu"
                        })
                        break
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Package not found or dpkg not available
            continue
    
    return packages


def generate_sbom() -> Dict[str, Any]:
    """Generate complete SBOM document."""
    git_info = get_git_info()
    python_packages = get_python_packages()
    system_packages = get_system_packages()
    
    # Read project metadata
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"
    
    project_name = "customer-churn-predictor"
    project_version = "1.0.0"
    
    # Try to extract from pyproject.toml
    if pyproject_path.exists():
        try:
            import tomllib
            with open(pyproject_path, "rb") as f:
                pyproject = tomllib.load(f)
                project_name = pyproject.get("project", {}).get("name", project_name)
                project_version = pyproject.get("project", {}).get("version", project_version)
        except Exception:
            pass
    
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "serialNumber": f"urn:uuid:{project_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now().isoformat() + "Z",
            "tools": [
                {
                    "vendor": "Terragon Labs",
                    "name": "SBOM Generator",
                    "version": "1.0.0"
                }
            ],
            "component": {
                "type": "application",
                "bom-ref": project_name,
                "name": project_name,
                "version": project_version,
                "description": "Production-ready ML system for customer churn prediction with comprehensive MLOps practices",
                "licenses": [
                    {
                        "license": {
                            "id": "Apache-2.0"
                        }
                    }
                ],
                "purl": f"pkg:github/danieleschmidt/{project_name}@{project_version}",
                "externalReferences": [
                    {
                        "type": "vcs",
                        "url": git_info["repository_url"]
                    },
                    {
                        "type": "build-meta",
                        "url": f"{git_info['repository_url']}/commit/{git_info['commit_hash']}"
                    }
                ]
            },
            "properties": [
                {
                    "name": "git.commit.hash",
                    "value": git_info["commit_hash"]
                },
                {
                    "name": "git.commit.date", 
                    "value": git_info["commit_date"]
                },
                {
                    "name": "build.date",
                    "value": datetime.now().isoformat() + "Z"
                }
            ]
        },
        "components": []
    }
    
    # Add Python packages
    for pkg in python_packages:
        component = {
            "type": "library",
            "bom-ref": f"python-{pkg['name']}@{pkg['version']}",
            "name": pkg["name"],
            "version": pkg["version"],
            "purl": f"pkg:pypi/{pkg['name']}@{pkg['version']}",
            "scope": "required"
        }
        
        if pkg.get("homepage"):
            component["externalReferences"] = [
                {
                    "type": "website",
                    "url": pkg["homepage"]
                }
            ]
        
        if pkg.get("author"):
            component["author"] = pkg["author"]
            
        sbom["components"].append(component)
    
    # Add system packages
    for pkg in system_packages:
        component = {
            "type": "library",
            "bom-ref": f"system-{pkg['name']}@{pkg['version']}",
            "name": pkg["name"],
            "version": pkg["version"],
            "scope": "required"
        }
        sbom["components"].append(component)
    
    return sbom


def main():
    """Main function to generate and save SBOM."""
    print("Generating Software Bill of Materials (SBOM)...")
    
    sbom = generate_sbom()
    
    # Save to file
    output_path = Path("sbom.json")
    with open(output_path, "w") as f:
        json.dump(sbom, f, indent=2, sort_keys=True)
    
    print(f"SBOM generated successfully: {output_path}")
    print(f"Total components: {len(sbom['components'])}")
    
    # Print summary
    python_count = sum(1 for c in sbom['components'] if c.get('purl', '').startswith('pkg:pypi/'))
    system_count = len(sbom['components']) - python_count
    
    print(f"  - Python packages: {python_count}")
    print(f"  - System packages: {system_count}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())