#!/usr/bin/env python3
"""
Docker integration test runner.

This script runs comprehensive Docker integration tests including:
- Container build and startup tests
- Health check integration tests  
- Environment variable validation
- Volume mount and permission tests
- Multi-service orchestration tests

Usage:
    python scripts/run_docker_tests.py [--live] [--cleanup] [--verbose]
    
    --live      Run live container tests (requires Docker)
    --cleanup   Clean up test containers and images after run
    --verbose   Enable verbose output
"""

import argparse
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_config import get_logger

logger = get_logger(__name__)


class DockerTestRunner:
    """Docker integration test runner with comprehensive validation."""
    
    def __init__(self, verbose: bool = False, cleanup: bool = True):
        self.verbose = verbose
        self.cleanup = cleanup
        self.test_results: Dict[str, Any] = {}
        
    def run_command(self, cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a command with logging."""
        if self.verbose:
            logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                check=False
            )
            
            if self.verbose and result.stdout:
                logger.info(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.warning(f"STDERR: {result.stderr}")
                
            return result
        except Exception as e:
            logger.error(f"Command failed: {e}")
            raise
    
    def check_docker_available(self) -> bool:
        """Check if Docker is available and running."""
        try:
            result = self.run_command(['docker', 'version'])
            if result.returncode == 0:
                logger.info("Docker is available and running")
                return True
            else:
                logger.error("Docker is not running")
                return False
        except Exception as e:
            logger.error(f"Docker availability check failed: {e}")
            return False
    
    def check_docker_compose_available(self) -> bool:
        """Check if Docker Compose is available."""
        try:
            result = self.run_command(['docker-compose', '--version'])
            if result.returncode == 0:
                logger.info("Docker Compose is available")
                return True
            else:
                logger.error("Docker Compose is not available")
                return False
        except Exception as e:
            logger.error(f"Docker Compose availability check failed: {e}")
            return False
    
    def run_static_tests(self) -> bool:
        """Run static Docker configuration tests."""
        logger.info("Running static Docker configuration tests...")
        
        try:
            # Run the static tests
            result = self.run_command([
                'python', '-m', 'pytest', 
                'tests/test_docker_integration.py',
                '-v' if self.verbose else '-q',
                '--tb=short'
            ])
            
            success = result.returncode == 0
            self.test_results['static_tests'] = {
                'success': success,
                'output': result.stdout,
                'errors': result.stderr
            }
            
            if success:
                logger.info("Static Docker tests passed")
            else:
                logger.error("Static Docker tests failed")
                
            return success
            
        except Exception as e:
            logger.error(f"Static tests failed: {e}")
            self.test_results['static_tests'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def run_live_tests(self) -> bool:
        """Run live Docker container tests."""
        if not self.check_docker_available():
            logger.warning("Skipping live tests - Docker not available")
            return True
            
        logger.info("Running live Docker container tests...")
        
        try:
            # Run the live tests
            result = self.run_command([
                'python', '-m', 'pytest',
                'tests/test_docker_health_check_live.py',
                '-v' if self.verbose else '-q',
                '--tb=short',
                '-x'  # Stop on first failure for faster feedback
            ])
            
            success = result.returncode == 0
            self.test_results['live_tests'] = {
                'success': success,
                'output': result.stdout,
                'errors': result.stderr
            }
            
            if success:
                logger.info("Live Docker tests passed")
            else:
                logger.error("Live Docker tests failed")
                
            return success
            
        except Exception as e:
            logger.error(f"Live tests failed: {e}")
            self.test_results['live_tests'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def validate_docker_compose(self) -> bool:
        """Validate Docker Compose configuration."""
        if not self.check_docker_compose_available():
            logger.warning("Skipping compose validation - Docker Compose not available")
            return True
            
        logger.info("Validating Docker Compose configuration...")
        
        try:
            # Validate compose file syntax
            result = self.run_command([
                'docker-compose', '-f', 'docker-compose.yml', 'config'
            ])
            
            success = result.returncode == 0
            self.test_results['compose_validation'] = {
                'success': success,
                'output': result.stdout,
                'errors': result.stderr
            }
            
            if success:
                logger.info("Docker Compose configuration is valid")
            else:
                logger.error("Docker Compose configuration validation failed")
                
            return success
            
        except Exception as e:
            logger.error(f"Compose validation failed: {e}")
            self.test_results['compose_validation'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_build_process(self) -> bool:
        """Test Docker image build process."""
        if not self.check_docker_available():
            logger.warning("Skipping build test - Docker not available")
            return True
            
        logger.info("Testing Docker image build process...")
        
        try:
            # Build the production image
            result = self.run_command([
                'docker', 'build', 
                '--target', 'production',
                '--tag', 'churn-predictor:test-build',
                '.'
            ])
            
            success = result.returncode == 0
            self.test_results['build_test'] = {
                'success': success,
                'output': result.stdout,
                'errors': result.stderr
            }
            
            if success:
                logger.info("Docker build test passed")
                
                # Test development build too
                dev_result = self.run_command([
                    'docker', 'build',
                    '--target', 'development', 
                    '--tag', 'churn-predictor:test-dev',
                    '.'
                ])
                
                if dev_result.returncode == 0:
                    logger.info("Development build also passed")
                else:
                    logger.warning("Development build failed but production passed")
                    
            else:
                logger.error("Docker build test failed")
                
            return success
            
        except Exception as e:
            logger.error(f"Build test failed: {e}")
            self.test_results['build_test'] = {
                'success': False,
                'error': str(e)
            }
            return False
        finally:
            if self.cleanup:
                self.cleanup_test_images()
    
    def cleanup_test_images(self):
        """Clean up test images."""
        test_images = [
            'churn-predictor:test-build',
            'churn-predictor:test-dev',
            'churn-predictor:test-health'
        ]
        
        for image in test_images:
            try:
                self.run_command(['docker', 'rmi', image, '-f'])
                if self.verbose:
                    logger.info(f"Cleaned up image: {image}")
            except:
                pass
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() 
                          if result.get('success', False))
        
        report = {
            'timestamp': time.time(),
            'summary': {
                'total_test_suites': total_tests,
                'passed_test_suites': passed_tests,
                'failed_test_suites': total_tests - passed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            'results': self.test_results,
            'environment': {
                'docker_available': self.check_docker_available(),
                'docker_compose_available': self.check_docker_compose_available()
            }
        }
        
        return report
    
    def run_all_tests(self, include_live: bool = False) -> bool:
        """Run all Docker integration tests."""
        logger.info("Starting comprehensive Docker integration tests...")
        
        all_passed = True
        
        # Run static tests (always)
        if not self.run_static_tests():
            all_passed = False
        
        # Validate Docker Compose
        if not self.validate_docker_compose():
            all_passed = False
        
        # Test build process
        if not self.test_build_process():
            all_passed = False
        
        # Run live tests if requested and Docker is available
        if include_live:
            if not self.run_live_tests():
                all_passed = False
        
        # Generate and display report
        report = self.generate_report()
        
        logger.info("=" * 60)
        logger.info("DOCKER INTEGRATION TEST REPORT")
        logger.info("=" * 60)
        logger.info(f"Test Suites: {report['summary']['passed_test_suites']}/{report['summary']['total_test_suites']} passed")
        logger.info(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
            
            if not result.get('success', False) and 'error' in result:
                logger.error(f"    Error: {result['error']}")
        
        logger.info("=" * 60)
        
        if all_passed:
            logger.info("🎉 All Docker integration tests passed!")
        else:
            logger.error("❌ Some Docker integration tests failed")
        
        return all_passed


def main():
    """Main entry point for Docker test runner."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive Docker integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--live',
        action='store_true',
        help='Run live container tests (requires Docker)'
    )
    
    parser.add_argument(
        '--cleanup',
        action='store_true',
        default=True,
        help='Clean up test containers and images after run (default: True)'
    )
    
    parser.add_argument(
        '--no-cleanup',
        action='store_false',
        dest='cleanup',
        help='Do not clean up test containers and images'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--report-file',
        help='Save detailed report to JSON file'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize test runner
    runner = DockerTestRunner(verbose=args.verbose, cleanup=args.cleanup)
    
    try:
        # Run all tests
        success = runner.run_all_tests(include_live=args.live)
        
        # Save report if requested
        if args.report_file:
            report = runner.generate_report()
            with open(args.report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Detailed report saved to: {args.report_file}")
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Test run interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test run failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()