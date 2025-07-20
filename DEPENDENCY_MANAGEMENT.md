# Dependency Management Guide

## Overview
This project uses a two-tier dependency management system for maximum reproducibility and flexibility:

1. **requirements.txt** - High-level dependencies with pinned major versions
2. **requirements.lock** - Complete dependency tree with exact versions
3. **requirements-dev.txt** - Development-specific dependencies
4. **requirements-dev.lock** - Complete development dependency tree

## File Structure

### Production Dependencies
- `requirements.txt` - Primary dependency declarations
- `requirements.lock` - Complete lockfile with all transitive dependencies

### Development Dependencies  
- `requirements-dev.txt` - Development tool dependencies
- `requirements-dev.lock` - Complete development lockfile

## Usage

### For Development
```bash
# Install development dependencies (includes production)
pip install -r requirements-dev.lock

# Run tests
pytest

# Code formatting
black .

# Type checking
mypy src/
```

### For Production/CI
```bash
# Install exact production dependencies
pip install -r requirements.lock

# Run application
python scripts/run_pipeline.py
```

### For Docker
```dockerfile
# Use lockfiles for reproducible builds
COPY requirements.lock .
RUN pip install --no-deps -r requirements.lock
```

## Updating Dependencies

### 1. Adding New Dependencies
1. Add to `requirements.txt` or `requirements-dev.txt` with version constraints
2. Regenerate lockfiles (see below)
3. Test thoroughly
4. Commit both the base file and lockfile changes

### 2. Updating Existing Dependencies
1. Update version in `requirements.txt` or `requirements-dev.txt`
2. Regenerate lockfiles
3. Test for compatibility issues
4. Update any breaking changes in code
5. Commit changes

### 3. Regenerating Lockfiles
Currently manual process (automated tool coming):

```bash
# Create clean environment
python -m venv temp_env
source temp_env/bin/activate  # Linux/Mac
# or
temp_env\Scripts\activate  # Windows

# Install production dependencies
pip install -r requirements.txt

# Generate production lockfile
pip freeze > requirements.lock

# Install dev dependencies
pip install -r requirements-dev.txt

# Generate dev lockfile  
pip freeze > requirements-dev.lock

# Clean up
deactivate
rm -rf temp_env
```

## Security Considerations

### Dependency Scanning
- All dependencies should be scanned for known vulnerabilities
- Use `pip-audit` or similar tools regularly
- Monitor security advisories for used packages

### Version Pinning Benefits
1. **Reproducibility** - Identical builds across environments
2. **Security** - Known versions without surprise updates
3. **Debugging** - Consistent environment for issue reproduction
4. **Compliance** - Audit trail of exact software versions

### Version Pinning Risks
1. **Security** - Must manually update for security patches
2. **Maintenance** - Regular updates required to avoid technical debt
3. **Compatibility** - May miss bug fixes in patch releases

## Best Practices

### 1. Regular Updates
- Review and update dependencies monthly
- Prioritize security updates immediately
- Test updates in development first

### 2. Semantic Versioning Awareness
- Major versions: Breaking changes expected
- Minor versions: New features, backward compatible
- Patch versions: Bug fixes, backward compatible

### 3. Dependency Hygiene
- Remove unused dependencies promptly
- Prefer packages with active maintenance
- Avoid dependencies with known security issues
- Document any pinned versions due to compatibility issues

### 4. Testing Strategy
- Run full test suite after dependency updates
- Test in production-like environment
- Monitor application behavior after deployment
- Have rollback plan ready

## Troubleshooting

### Common Issues
1. **Version Conflicts**: Use `pip check` to identify conflicts
2. **Missing Dependencies**: Ensure lockfile includes all transitive deps
3. **Build Failures**: Check for platform-specific dependencies
4. **Import Errors**: Verify package names and versions

### Resolution Steps
1. Check error messages for specific version conflicts
2. Review dependency tree with `pip show <package>`
3. Use virtual environments to isolate issues
4. Consider using `pip-tools` for more sophisticated resolution

## Future Improvements
- [ ] Automated lockfile generation with pip-tools
- [ ] Pre-commit hooks for dependency validation
- [ ] Automated security scanning in CI/CD
- [ ] Dependency update automation with testing
- [ ] Multi-platform lockfile support