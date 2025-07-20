# Development Backlog - Customer Churn Predictor MLOps

**Last Updated**: 2025-07-20  
**Prioritization Method**: WSJF (Weighted Shortest Job First)

## Scoring Methodology
**WSJF Score = (Business Value + Time Criticality + Risk Reduction) / Job Size**
- All factors scored 1-10
- Higher WSJF = Higher Priority

---

## CRITICAL PRIORITY (WSJF > 2.0)

### 1. Fix Performance Issue - DataFrame.iterrows() (WSJF: 3.5)
**Files**: `scripts/run_prediction.py`
- **Business Value**: 7 (user experience impact)
- **Time Criticality**: 7 (scalability blocker)
- **Risk Reduction**: 8 (prevents performance degradation)
- **Job Size**: 2 (simple vectorization)
- **Description**: Replace inefficient iterrows() with vectorized operations
- **Acceptance Criteria**: Batch predictions run 10x faster

### 2. Implement Centralized Logging System (WSJF: 2.67)
**Files**: All modules with print statements
- **Business Value**: 8 (observability)
- **Time Criticality**: 8 (production requirement)
- **Risk Reduction**: 8 (debugging capability)
- **Job Size**: 3 (structured refactor)
- **Description**: Replace all print statements with proper logging
- **Acceptance Criteria**: All outputs use Python logging with appropriate levels

### 3. Add Dependency Version Pinning (WSJF: 2.5)
**Files**: `requirements.txt`, new `requirements.lock`
- **Business Value**: 5 (reproducibility)
- **Time Criticality**: 7 (deployment reliability)
- **Risk Reduction**: 8 (prevents version conflicts)
- **Job Size**: 2 (tooling setup)
- **Description**: Pin exact dependency versions for reproducible builds
- **Acceptance Criteria**: All dependencies locked to specific versions

### 4. ✅ Fix Bare Exception Handling (WSJF: 2.25) - COMPLETED
**Files**: `predict_churn.py`, `monitor_performance.py`
- **Business Value**: 9 (error handling)
- **Time Criticality**: 8 (production stability)
- **Risk Reduction**: 9 (prevents masked errors)
- **Job Size**: 4 (careful refactoring)
- **Description**: Replace bare except clauses with specific exception handling
- **Acceptance Criteria**: All exceptions caught specifically with proper error propagation
- **Completed**: 2025-07-20 - Replaced all bare Exception handlers with specific exception types

### 5. ✅ Add Missing Type Hints (WSJF: 2.0) - COMPLETED
**Files**: `predict_churn.py`, `cli.py`, other modules
- **Business Value**: 6 (code quality)
- **Time Criticality**: 5 (developer experience)
- **Risk Reduction**: 7 (static analysis benefits)
- **Job Size**: 3 (systematic addition)
- **Description**: Add comprehensive type hints throughout codebase
- **Acceptance Criteria**: All functions have complete type annotations
- **Completed**: 2025-07-20 - Added type hints to all core modules and CLI commands

### 6. ✅ Environment Variable Validation (WSJF: 2.0) - COMPLETED
**Files**: `monitor_performance.py`, `config.py`
- **Business Value**: 6 (configuration safety)
- **Time Criticality**: 7 (runtime stability)
- **Risk Reduction**: 8 (prevents config errors)
- **Job Size**: 3 (validation logic)
- **Description**: Add proper validation for all environment variables
- **Acceptance Criteria**: All env vars validated with defaults and error handling
- **Completed**: 2025-07-20 - Implemented comprehensive environment variable validation framework

---

## HIGH PRIORITY (WSJF 1.5-2.0)

### 7. Implement Input Validation Framework (WSJF: 1.8)
**Files**: All modules accepting file paths or user input
- **Business Value**: 9 (security)
- **Time Criticality**: 7 (vulnerability mitigation)
- **Risk Reduction**: 9 (prevents attacks)
- **Job Size**: 5 (comprehensive validation)
- **Description**: Add validation for file paths, data types, ranges
- **Acceptance Criteria**: All user inputs validated, path traversal prevented

### 8. Refactor Hardcoded File Paths (WSJF: 1.75)
**Files**: `constants.py`, various modules
- **Business Value**: 7 (portability)
- **Time Criticality**: 6 (deployment flexibility)
- **Risk Reduction**: 8 (environment independence)
- **Job Size**: 4 (configuration refactor)
- **Description**: Replace hardcoded paths with configurable options
- **Acceptance Criteria**: All paths configurable via environment variables

### 9. Add Security for File Operations (WSJF: 1.6)
**Files**: All modules with file I/O
- **Business Value**: 8 (security)
- **Time Criticality**: 7 (vulnerability fix)
- **Risk Reduction**: 9 (prevents path traversal)
- **Job Size**: 5 (security implementation)
- **Description**: Implement path sanitization and access control
- **Acceptance Criteria**: No path traversal vulnerabilities remain

### 10. Extract MLflow Utilities (WSJF: 1.5)
**Files**: `predict_churn.py`, `monitor_performance.py`
- **Business Value**: 6 (maintainability)
- **Time Criticality**: 5 (technical debt)
- **Risk Reduction**: 7 (consistency)
- **Job Size**: 4 (utility extraction)
- **Description**: Create shared MLflow download utilities
- **Acceptance Criteria**: Single source of truth for MLflow operations

### 11. Implement CI/CD Pipeline (WSJF: 1.5)
**Files**: `.github/workflows/main.yml`
- **Business Value**: 9 (automation)
- **Time Criticality**: 6 (process improvement)
- **Risk Reduction**: 8 (quality gates)
- **Job Size**: 6 (complex setup)
- **Description**: Add GitHub Actions with linting, testing, security scanning
- **Acceptance Criteria**: Full CI pipeline with quality gates
- **Note**: Requires manual implementation due to workflow permissions

---

## MEDIUM PRIORITY (WSJF 1.0-1.5)

### 12. Add Missing Docstrings (WSJF: 1.67)
**Files**: `predict_churn.py`, `cli.py`
- **Business Value**: 5 (documentation)
- **Time Criticality**: 4 (maintainability)
- **Risk Reduction**: 6 (knowledge transfer)
- **Job Size**: 3 (documentation writing)

### 13. Implement Data Validation (WSJF: 1.33)
**Files**: All data processing modules
- **Business Value**: 8 (data integrity)
- **Time Criticality**: 6 (production quality)
- **Risk Reduction**: 8 (prevents bad predictions)
- **Job Size**: 6 (schema implementation)

### 14. Add Edge Case Test Coverage (WSJF: 1.4)
**Files**: All test modules
- **Business Value**: 7 (quality assurance)
- **Time Criticality**: 5 (regression prevention)
- **Risk Reduction**: 8 (error detection)
- **Job Size**: 5 (comprehensive testing)

---

## COMPLETED ITEMS

- ✅ Initial project setup and structure
- ✅ Basic MLflow integration
- ✅ Core preprocessing pipeline
- ✅ Model training functionality
- ✅ Basic CLI interface with Typer
- ✅ Configuration management system
- ✅ Basic unit test structure
- ✅ Fix Performance Issue - DataFrame.iterrows() (WSJF: 3.5) - 2025-07-20
- ✅ Implement Centralized Logging System (WSJF: 2.67) - 2025-07-20
- ✅ Add Dependency Version Pinning (WSJF: 2.5) - 2025-07-20
- ✅ Fix Bare Exception Handling (WSJF: 2.25) - 2025-07-20
- ✅ Add Missing Type Hints (WSJF: 2.0) - 2025-07-20
- ✅ Environment Variable Validation (WSJF: 2.0) - 2025-07-20

---

## RISK ASSESSMENT

### High Risk Items:
1. **Performance bottleneck** in prediction pipeline (Item #1)
2. **Security vulnerabilities** in file handling (Items #7, #9)
3. **Production reliability** from poor error handling (Item #4)
4. **Deployment issues** from missing dependency pinning (Item #3)

### Dependencies:
- Items #2, #4, #5 should be completed before major refactoring
- Item #11 requires manual workflow file changes
- Items #7, #9 are security-critical and should be prioritized

---

*Backlog maintained using autonomous development principles*  
*Next review: After each major milestone completion*