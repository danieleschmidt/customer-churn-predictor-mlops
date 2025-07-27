# GitHub Workflows for Manual Addition

Due to GitHub App permission restrictions, these workflow files need to be manually added to `.github/workflows/` by a repository administrator.

## Files to Add

1. **security.yml** → `.github/workflows/security.yml`
   - Weekly security scanning with Safety, Bandit, and Semgrep
   - Automated security report generation
   - Runs on push, PR, and weekly schedule

2. **dependency-update.yml** → `.github/workflows/dependency-update.yml`
   - Weekly automated dependency updates
   - Security vulnerability checks during updates
   - Automated PR creation for dependency updates

3. **release.yml** → `.github/workflows/release.yml`
   - Automated release process triggered by version tags
   - Changelog generation from git history
   - Package building and artifact publishing

## How to Add

1. Copy each `.yml` file to `.github/workflows/` directory
2. Commit and push the changes
3. The workflows will automatically become active

## Permissions Required

These workflows require the following GitHub permissions:
- `contents: read` - To checkout code
- `security-events: write` - For security scanning results
- `pull-requests: write` - For creating dependency update PRs
- `releases: write` - For creating releases

Most of these permissions are included by default with `GITHUB_TOKEN`.