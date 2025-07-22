# Development Backlog - Customer Churn Predictor MLOps

**Last Updated**: 2025-07-20  
**Prioritization Method**: WSJF (Weighted Shortest Job First)

## Scoring Methodology
**WSJF Score = (Business Value + Time Criticality + Risk Reduction) / Job Size**
- All factors scored 1-10
- Higher WSJF = Higher Priority

---

## CRITICAL PRIORITY (WSJF > 2.0)

### 1. ✅ Replace Print Statements in Utility Scripts (WSJF: 2.4) - COMPLETED
**Files**: `verify_logging.py`, `validate_dependencies.py`, `migrate_logging.py`, `generate_lockfile.py`
- **Business Value**: 8 (observability consistency)
- **Time Criticality**: 8 (operational standards)
- **Risk Reduction**: 8 (proper log management)
- **Job Size**: 4 (systematic replacement)
- **Description**: Replace all print statements with proper logging in utility scripts
- **Acceptance Criteria**: All utility scripts use structured logging with appropriate levels
- **Completed**: 2025-07-22 - Replaced all print statements with structured logging using get_logger() pattern

### 2. ✅ Fix Performance Issue - DataFrame.iterrows() (WSJF: 3.5) - COMPLETED
**Files**: `scripts/run_prediction.py`
- **Business Value**: 7 (user experience impact)
- **Time Criticality**: 7 (scalability blocker)
- **Risk Reduction**: 8 (prevents performance degradation)
- **Job Size**: 2 (simple vectorization)
- **Description**: Replace inefficient iterrows() with vectorized operations
- **Acceptance Criteria**: Batch predictions run 10x faster
- **Completed**: Previously completed but fallback code still exists

### 3. ✅ Implement Centralized Logging System (WSJF: 2.67) - PARTIALLY COMPLETED
**Files**: All modules with print statements
- **Business Value**: 8 (observability)
- **Time Criticality**: 8 (production requirement)
- **Risk Reduction**: 8 (debugging capability)
- **Job Size**: 3 (structured refactor)
- **Description**: Replace all print statements with proper logging
- **Acceptance Criteria**: All outputs use Python logging with appropriate levels
- **Note**: Main modules completed, utility scripts still need conversion

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

### 4. ✅ Remove Global State in Path Configuration (WSJF: 2.1) - COMPLETED
**Files**: `src/path_config.py`, `src/constants.py`, `src/config.py`, `scripts/run_preprocessing.py`
- **Business Value**: 7 (code quality)
- **Time Criticality**: 6 (maintainability)
- **Risk Reduction**: 9 (prevents race conditions)
- **Job Size**: 3 (refactor to instance-based)
- **Description**: Replace global _global_config variable with proper instance management
- **Acceptance Criteria**: No global state, thread-safe configuration management
- **Completed**: 2025-07-22 - Eliminated global state, implemented dependency injection pattern with PathConfig instances, updated all modules to support both legacy and new approaches, added comprehensive thread safety tests, updated documentation with migration guide and best practices

---

## HIGH PRIORITY (WSJF 1.5-2.0)

### 7. ✅ Implement Input Validation Framework (WSJF: 1.8) - COMPLETED
**Files**: All modules accepting file paths or user input
- **Business Value**: 9 (security)
- **Time Criticality**: 7 (vulnerability mitigation)
- **Risk Reduction**: 9 (prevents attacks)
- **Job Size**: 5 (comprehensive validation)
- **Description**: Add validation for file paths, data types, ranges
- **Acceptance Criteria**: All user inputs validated, path traversal prevented
- **Completed**: 2025-07-21 - Implemented comprehensive validation framework with PathValidator, DataValidator, MLValidator classes and integrated into core modules

### 8. ✅ Refactor Hardcoded File Paths (WSJF: 1.75) - COMPLETED
**Files**: `constants.py`, various modules
- **Business Value**: 7 (portability)
- **Time Criticality**: 6 (deployment flexibility)
- **Risk Reduction**: 8 (environment independence)
- **Job Size**: 4 (configuration refactor)
- **Description**: Replace hardcoded paths with configurable options
- **Acceptance Criteria**: All paths configurable via environment variables
- **Completed**: 2025-07-21 - Implemented comprehensive path configuration system with PathConfig class, environment variable support, deployment examples, and backwards compatibility

### 9. ✅ Add Security for File Operations (WSJF: 1.6) - COMPLETED
**Files**: All modules with file I/O
- **Business Value**: 8 (security)
- **Time Criticality**: 7 (vulnerability fix)
- **Risk Reduction**: 9 (prevents path traversal)
- **Job Size**: 5 (security implementation)
- **Description**: Implement path sanitization and access control
- **Acceptance Criteria**: No path traversal vulnerabilities remain
- **Completed**: 2025-07-22 - Enhanced validation.py with secure file I/O functions (safe_read_csv, safe_write_csv, safe_read_json, safe_write_json, safe_read_text, safe_write_text), updated core modules to use secure operations. Final security hardening completed by replacing remaining insecure json.dump operations in predict_churn.py.

### 10. ✅ Replace Generic Exception Handlers (WSJF: 1.8) - COMPLETED
**Files**: `validation.py`, `logging_config.py`, utility scripts
- **Business Value**: 6 (debugging capability)
- **Time Criticality**: 6 (production support)
- **Risk Reduction**: 7 (error transparency)
- **Job Size**: 3 (systematic replacement)
- **Description**: Replace generic 'except Exception' handlers with specific exception types
- **Acceptance Criteria**: All exception handlers catch specific exception types with appropriate error handling
- **Completed**: 2025-07-22 - Replaced 9 generic exception handlers with specific types (FileNotFoundError, PermissionError, UnicodeDecodeError, json.JSONDecodeError, pd.errors.*, subprocess.CalledProcessError, etc.). Added comprehensive test coverage.

### 11. ✅ Add Missing Test Coverage (WSJF: 1.6) - COMPLETED
**Files**: `src/cli.py`, `src/constants.py`, `src/env_config.py`, `src/logging_config.py`
- **Business Value**: 7 (quality assurance)
- **Time Criticality**: 5 (regression prevention)
- **Risk Reduction**: 7 (coverage gaps)
- **Job Size**: 4 (multiple test files)
- **Description**: Add comprehensive test coverage for untested core modules
- **Acceptance Criteria**: 80%+ test coverage for all core modules with unit and integration tests
- **Completed**: 2025-07-22 - Created comprehensive test suites for all untested core modules: test_cli.py (17 test classes, 25+ test methods covering CLI commands, parameter handling, validation), test_constants.py (7 test classes, 31 test methods covering legacy constants, factory functions, environment handling), test_env_config.py (11 test classes, 34 test methods covering validation, error handling, configuration loading), enhanced test_logging_system.py with 4 additional test classes covering MLflow logging, decorators, auto-configuration

### 12. ✅ Extract MLflow Utilities (WSJF: 1.5) - COMPLETED
**Files**: `predict_churn.py`, `monitor_performance.py`, `src/mlflow_utils.py`, `tests/test_mlflow_utils.py`
- **Business Value**: 6 (maintainability)
- **Time Criticality**: 5 (technical debt)
- **Risk Reduction**: 7 (consistency)
- **Job Size**: 4 (utility extraction)
- **Description**: Create shared MLflow download utilities
- **Acceptance Criteria**: Single source of truth for MLflow operations
- **Completed**: 2025-07-22 - Created centralized MLflow utilities module with shared functions for model/artifact downloading, evaluation metrics logging, and MLflowArtifactManager context manager. Refactored predict_churn.py and monitor_performance.py to use shared utilities, eliminating code duplication. Added comprehensive test coverage with 11 test classes. Made MLflow imports conditional for better compatibility.

### 13. Implement CI/CD Pipeline (WSJF: 1.5)
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

### 14. ✅ Add Missing Docstrings (WSJF: 1.67) - COMPLETED
**Files**: `predict_churn.py`, `cli.py`, `scripts/run_*.py`
- **Business Value**: 5 (documentation)
- **Time Criticality**: 4 (maintainability)
- **Risk Reduction**: 6 (knowledge transfer)
- **Job Size**: 3 (documentation writing)
- **Description**: Add comprehensive docstrings to core functions and CLI entry points
- **Acceptance Criteria**: All public functions and CLI commands have detailed docstrings
- **Completed**: 2025-07-22 - Added comprehensive Google/NumPy style docstrings to all 7 CLI functions and 5 main() functions in scripts directory. Documentation includes parameter descriptions, usage examples, command-line arguments, and 400+ lines of detailed documentation covering all entry points.

### 15. Implement Data Validation (WSJF: 1.33)
**Files**: All data processing modules
- **Business Value**: 8 (data integrity)
- **Time Criticality**: 6 (production quality)
- **Risk Reduction**: 8 (prevents bad predictions)
- **Job Size**: 6 (schema implementation)

### 16. Add Edge Case Test Coverage (WSJF: 1.4)
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
- ✅ Implement Input Validation Framework (WSJF: 1.8) - 2025-07-21
- ✅ Refactor Hardcoded File Paths (WSJF: 1.75) - 2025-07-21
- ✅ Replace Print Statements in Utility Scripts (WSJF: 2.4) - 2025-07-22
- ✅ Add Security for File Operations (WSJF: 1.6) - 2025-07-22
- ✅ Complete Security Hardening in predict_churn.py (WSJF: 2.0) - 2025-07-22
- ✅ Replace Generic Exception Handlers (WSJF: 1.8) - 2025-07-22
- ✅ Remove Global State in Path Configuration (WSJF: 2.1) - 2025-07-22
- ✅ Add Missing Test Coverage (WSJF: 1.6) - 2025-07-22
- ✅ Extract MLflow Utilities (WSJF: 1.5) - 2025-07-22
- ✅ Add Missing Docstrings (WSJF: 1.67) - 2025-07-22

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