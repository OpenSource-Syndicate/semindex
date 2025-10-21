"""
Testing Framework for semindex AI CLI
Generates and runs tests for developed components
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .ai_implementation_assistant import AIImplementationAssistant
from .project_planner import ProjectPlan, ProjectComponent


class TestingFramework:
    """Generates and manages tests for AI-developed components"""
    
    def __init__(self, project_plan: ProjectPlan, index_dir: str):
        self.project_plan = project_plan
        self.index_dir = index_dir
        self.ai_assistant = AIImplementationAssistant(index_dir, config_path=None)
        
        # Create a directory for tests
        self.test_dir = Path("tests")
        self.test_dir.mkdir(exist_ok=True)
    
    def generate_tests_for_component(self, component: ProjectComponent, 
                                   test_framework: str = "pytest") -> str:
        """Generate tests for a specific component"""
        # Get the component implementation for context
        implementation_code = ""
        try:
            with open(component.file_path, 'r', encoding='utf-8') as f:
                implementation_code = f.read()
        except FileNotFoundError:
            print(f"Warning: Component file not found at {component.file_path}")
        
        # Generate tests based on the component type
        if component.component_type == "function":
            return self._generate_function_tests(component, implementation_code, test_framework)
        elif component.component_type == "class":
            return self._generate_class_tests(component, implementation_code, test_framework)
        else:
            return self._generate_generic_tests(component, implementation_code, test_framework)
    
    def _generate_function_tests(self, component: ProjectComponent, 
                               implementation: str, framework: str) -> str:
        """Generate tests for a function component"""
        system_prompt = (
            f"You are a senior Python developer. Generate comprehensive unit tests using {framework} "
            f"for the provided function. Create tests for normal cases, edge cases, error conditions, "
            f"and boundary values. Follow best practices for testing."
        )
        
        user_prompt = f"Function implementation:\n{implementation}\n\nGenerate unit tests:"
        
        llm = self.ai_assistant._get_llm_instance()
        test_code = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1024
        )
        
        return test_code
    
    def _generate_class_tests(self, component: ProjectComponent, 
                            implementation: str, framework: str) -> str:
        """Generate tests for a class component"""
        system_prompt = (
            f"You are a senior Python developer. Generate comprehensive unit tests using {framework} "
            f"for the provided class. Test all methods, properties, initialization, and edge cases. "
            f"Follow best practices for testing object-oriented code."
        )
        
        user_prompt = f"Class implementation:\n{implementation}\n\nGenerate unit tests:"
        
        llm = self.ai_assistant._get_llm_instance()
        test_code = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1024
        )
        
        return test_code
    
    def _generate_generic_tests(self, component: ProjectComponent, 
                              implementation: str, framework: str) -> str:
        """Generate generic tests for other component types"""
        system_prompt = (
            f"You are a senior Python developer. Generate basic tests using {framework} "
            f"for the provided code component. Focus on the main functionality."
        )
        
        user_prompt = f"Component implementation:\n{implementation}\n\nGenerate basic tests:"
        
        llm = self.ai_assistant._get_llm_instance()
        test_code = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1024
        )
        
        return test_code
    
    def generate_all_tests(self, test_framework: str = "pytest") -> Dict[str, str]:
        """Generate tests for all components in the project"""
        test_files = {}
        
        for component in self.project_plan.components:
            # Generate tests for the component
            test_code = self.generate_tests_for_component(component, test_framework)
            
            # Determine the appropriate test file path
            test_file_path = self._determine_test_file_path(component, test_framework)
            
            # Store the test code
            test_files[test_file_path] = test_code
        
        return test_files
    
    def _determine_test_file_path(self, component: ProjectComponent, framework: str) -> str:
        """Determine the appropriate test file path for a component"""
        # Convert the component file path to a test file path
        component_path = Path(component.file_path)
        
        # Create a test file name based on the component file
        test_file_name = f"test_{component_path.name}"
        if not test_file_name.startswith("test_"):
            test_file_name = f"test_{test_file_name}"
        
        # Create test directory structure similar to source
        relative_path = component_path.parent
        test_path = self.test_dir / relative_path / test_file_name
        
        return str(test_path)
    
    def write_tests_to_files(self, test_files: Dict[str, str]) -> List[str]:
        """Write generated tests to appropriate files"""
        written_files = []
        
        for test_file_path, test_code in test_files.items():
            # Create the directory if it doesn't exist
            Path(test_file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write the test code to the file
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            written_files.append(test_file_path)
            print(f"Generated test file: {test_file_path}")
        
        return written_files
    
    def run_tests(self, test_file: Optional[str] = None) -> Dict[str, any]:
        """Run the generated tests and return results"""
        try:
            # Determine which tests to run
            if test_file:
                # Run specific test file
                cmd = [sys.executable, "-m", "pytest", test_file, "-v"]
            else:
                # Run all tests in the test directory
                cmd = [sys.executable, "-m", "pytest", str(self.test_dir), "-v"]
            
            # Execute the test command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            return {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd)
            }
        except Exception as e:
            return {
                "return_code": -1,
                "error": str(e),
                "command": " ".join(cmd) if 'cmd' in locals() else ""
            }
    
    def validate_test_coverage(self) -> Dict[str, any]:
        """Validate test coverage of the codebase"""
        try:
            # Install coverage if not already installed
            subprocess.run([sys.executable, "-m", "pip", "install", "coverage"], 
                          capture_output=True)
            
            # Run tests with coverage
            cmd = [sys.executable, "-m", "coverage", "run", "-m", "pytest", str(self.test_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Generate coverage report
                report_cmd = [sys.executable, "-m", "coverage", "report", "-m"]
                report_result = subprocess.run(report_cmd, capture_output=True, text=True)
                
                return {
                    "return_code": report_result.returncode,
                    "coverage_report": report_result.stdout,
                    "error": report_result.stderr if report_result.returncode != 0 else None
                }
            else:
                return {
                    "return_code": result.returncode,
                    "error": result.stderr
                }
        except Exception as e:
            return {
                "return_code": -1,
                "error": str(e)
            }
    
    def create_test_plan(self) -> str:
        """Create a test plan based on the project components"""
        system_prompt = (
            "You are a senior test engineer. Based on the provided list of components, "
            "create a comprehensive test plan that outlines what needs to be tested, "
            "test strategies for different components, and testing phases. "
            "Include unit tests, integration tests, and any specific testing requirements."
        )
        
        # Create a description of all components
        components_description = []
        for component in self.project_plan.components:
            components_description.append(
                f"- {component.component_type} '{component.name}' in {component.file_path}: {component.description}"
            )
        
        user_prompt = f"Project components:\n" + "\n".join(components_description) + "\n\nCreate a test plan:"
        
        llm = self.ai_assistant._get_llm_instance()
        test_plan = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1024
        )
        
        return test_plan