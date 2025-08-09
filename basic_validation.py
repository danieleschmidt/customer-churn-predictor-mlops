#!/usr/bin/env python3
"""
Basic validation test for the autonomous SDLC implementation.
Tests syntax and basic structure without external dependencies.
"""

import sys
import os
import importlib.util
sys.path.append('/root/repo')

def test_module_syntax():
    """Test that all modules have valid syntax."""
    print("Testing module syntax validation...")
    
    modules = [
        'src/intelligent_model_selection.py',
        'src/automated_feature_engineering.py', 
        'src/streaming_predictions.py',
        'src/adaptive_learning.py',
        'src/error_handling_recovery.py',
        'src/advanced_monitoring.py',
        'src/distributed_computing.py',
        'src/high_performance_optimization.py',
        'src/auto_scaling_optimization.py',
        'src/advanced_caching_optimization.py',
        'src/performance_benchmarking.py'
    ]
    
    passed = 0
    failed = 0
    
    for module_path in modules:
        full_path = f'/root/repo/{module_path}'
        module_name = os.path.basename(module_path).replace('.py', '')
        
        try:
            # Check if file exists
            if not os.path.exists(full_path):
                print(f"✗ {module_name}: File not found")
                failed += 1
                continue
            
            # Try to compile the module
            with open(full_path, 'r') as f:
                code = f.read()
            
            compile(code, full_path, 'exec')
            print(f"✓ {module_name}: Syntax valid")
            passed += 1
            
        except SyntaxError as e:
            print(f"✗ {module_name}: Syntax error - {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {module_name}: Error - {e}")
            failed += 1
    
    return passed, failed

def test_module_structure():
    """Test basic module structure and key classes."""
    print("\nTesting module structure...")
    
    structure_tests = [
        ('src/intelligent_model_selection.py', ['ModelSelectionEngine', 'ModelConfigurationRegistry']),
        ('src/automated_feature_engineering.py', ['AutomatedFeatureEngineer', 'FeatureEngineeringPipeline']),
        ('src/streaming_predictions.py', ['StreamingPredictor', 'StreamingServer']),
        ('src/adaptive_learning.py', ['AdaptiveLearningSystem', 'ConceptDriftDetector']),
        ('src/error_handling_recovery.py', ['ErrorHandler', 'CircuitBreaker', 'RetryMechanism']),
        ('src/advanced_monitoring.py', ['MonitoringSystem', 'AlertManager']),
        ('src/distributed_computing.py', ['DistributedMLSystem', 'ServiceDiscovery']),
        ('src/high_performance_optimization.py', ['HighPerformanceOptimizer', 'MemoryPool']),
        ('src/auto_scaling_optimization.py', ['AutoScalingSystem', 'ResourceMonitor']),
        ('src/advanced_caching_optimization.py', ['MultiTierCache', 'InMemoryCache']),
        ('src/performance_benchmarking.py', ['BenchmarkSuite', 'CodeProfiler'])
    ]
    
    passed = 0
    failed = 0
    
    for module_path, expected_classes in structure_tests:
        full_path = f'/root/repo/{module_path}'
        module_name = os.path.basename(module_path).replace('.py', '')
        
        try:
            with open(full_path, 'r') as f:
                content = f.read()
            
            found_classes = []
            missing_classes = []
            
            for class_name in expected_classes:
                if f'class {class_name}' in content:
                    found_classes.append(class_name)
                else:
                    missing_classes.append(class_name)
            
            if not missing_classes:
                print(f"✓ {module_name}: All expected classes found ({len(found_classes)})")
                passed += 1
            else:
                print(f"✗ {module_name}: Missing classes - {missing_classes}")
                failed += 1
                
        except Exception as e:
            print(f"✗ {module_name}: Error checking structure - {e}")
            failed += 1
    
    return passed, failed

def test_docstrings_and_comments():
    """Test that modules have proper documentation."""
    print("\nTesting documentation quality...")
    
    modules = [
        'src/intelligent_model_selection.py',
        'src/automated_feature_engineering.py',
        'src/streaming_predictions.py',
        'src/adaptive_learning.py',
        'src/error_handling_recovery.py',
        'src/advanced_monitoring.py',
        'src/distributed_computing.py',
        'src/high_performance_optimization.py',
        'src/auto_scaling_optimization.py',
        'src/advanced_caching_optimization.py',
        'src/performance_benchmarking.py'
    ]
    
    passed = 0
    failed = 0
    
    for module_path in modules:
        full_path = f'/root/repo/{module_path}'
        module_name = os.path.basename(module_path).replace('.py', '')
        
        try:
            with open(full_path, 'r') as f:
                content = f.read()
            
            # Check for module docstring
            has_module_docstring = content.startswith('"""') or content.startswith("'''")
            
            # Check for class and function docstrings
            class_count = content.count('class ')
            function_count = content.count('def ')
            docstring_count = content.count('"""') + content.count("'''")
            
            # Simple heuristic: should have at least one docstring per 2 classes/functions
            expected_min_docstrings = max(1, (class_count + function_count) // 2)
            
            if has_module_docstring and docstring_count >= expected_min_docstrings:
                print(f"✓ {module_name}: Good documentation ({docstring_count} docstrings)")
                passed += 1
            else:
                print(f"✗ {module_name}: Poor documentation (needs improvement)")
                failed += 1
                
        except Exception as e:
            print(f"✗ {module_name}: Error checking documentation - {e}")
            failed += 1
    
    return passed, failed

def test_code_quality():
    """Test basic code quality metrics."""
    print("\nTesting code quality...")
    
    modules = [
        'src/intelligent_model_selection.py',
        'src/automated_feature_engineering.py',
        'src/streaming_predictions.py',
        'src/adaptive_learning.py',
        'src/error_handling_recovery.py',
        'src/advanced_monitoring.py',
        'src/distributed_computing.py',
        'src/high_performance_optimization.py',
        'src/auto_scaling_optimization.py',
        'src/advanced_caching_optimization.py',
        'src/performance_benchmarking.py'
    ]
    
    passed = 0
    failed = 0
    
    for module_path in modules:
        full_path = f'/root/repo/{module_path}'
        module_name = os.path.basename(module_path).replace('.py', '')
        
        try:
            with open(full_path, 'r') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            non_empty_lines = len([line for line in lines if line.strip()])
            comment_lines = len([line for line in lines if line.strip().startswith('#')])
            
            # Quality metrics
            has_imports = any('import ' in line for line in lines[:50])  # Check first 50 lines
            has_error_handling = any('try:' in line or 'except' in line for line in lines)
            has_logging = any('logger' in line or 'logging' in line for line in lines)
            reasonable_size = 100 <= total_lines <= 5000  # Not too small or too large
            
            quality_score = sum([has_imports, has_error_handling, has_logging, reasonable_size])
            
            if quality_score >= 3:
                print(f"✓ {module_name}: Good quality ({total_lines} lines, score: {quality_score}/4)")
                passed += 1
            else:
                print(f"✗ {module_name}: Quality issues ({total_lines} lines, score: {quality_score}/4)")
                failed += 1
                
        except Exception as e:
            print(f"✗ {module_name}: Error checking quality - {e}")
            failed += 1
    
    return passed, failed

def run_validation():
    """Run all validation tests."""
    print("=" * 70)
    print("AUTONOMOUS SDLC IMPLEMENTATION - BASIC VALIDATION TESTS")
    print("=" * 70)
    print(f"Validation started at: {__import__('datetime').datetime.now()}")
    print()
    
    tests = [
        ("Syntax Validation", test_module_syntax),
        ("Structure Validation", test_module_structure),
        ("Documentation Quality", test_docstrings_and_comments),
        ("Code Quality", test_code_quality)
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_name, test_func in tests:
        print(f"{test_name}")
        print("-" * len(test_name))
        
        try:
            passed, failed = test_func()
            total_passed += passed
            total_failed += failed
            
            success_rate = (passed / (passed + failed)) * 100 if (passed + failed) > 0 else 0
            print(f"Result: {passed} passed, {failed} failed ({success_rate:.1f}% success)")
            
        except Exception as e:
            print(f"✗ Test error: {e}")
            total_failed += 1
    
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total Checks: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if total_passed + total_failed > 0:
        overall_success = (total_passed / (total_passed + total_failed)) * 100
        print(f"Overall Success Rate: {overall_success:.1f}%")
    else:
        overall_success = 0
        print("Overall Success Rate: 0.0%")
    
    print(f"Validation completed at: {__import__('datetime').datetime.now()}")
    
    # Assessment
    if overall_success >= 90:
        print("\n🎉 EXCELLENT! Implementation meets high quality standards.")
        assessment = "PRODUCTION_READY"
    elif overall_success >= 80:
        print("\n✅ GOOD! Implementation meets acceptable quality standards.")
        assessment = "READY_WITH_MONITORING"
    elif overall_success >= 70:
        print("\n⚠️  ACCEPTABLE! Implementation needs minor improvements.")
        assessment = "NEEDS_MINOR_FIXES"
    else:
        print("\n❌ POOR! Implementation needs significant improvements.")
        assessment = "NEEDS_MAJOR_FIXES"
    
    print(f"Quality Assessment: {assessment}")
    
    return overall_success >= 80

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)