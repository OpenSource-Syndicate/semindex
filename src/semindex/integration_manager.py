"""
Integration Manager for semindex AI CLI
Manages the integration of developed components into a cohesive system
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .ai_implementation_assistant import AIImplementationAssistant
from .project_planner import ProjectPlan, ProjectComponent


class IntegrationManager:
    """Manages the integration of developed components into a cohesive system"""
    
    def __init__(self, project_plan: ProjectPlan, index_dir: str):
        self.project_plan = project_plan
        self.index_dir = index_dir
        self.ai_assistant = AIImplementationAssistant(index_dir, config_path=None)
        
        # Create a directory for integration artifacts
        self.integration_dir = Path(".semindex/integration")
        self.integration_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_component_dependencies(self) -> Dict[str, List[str]]:
        """Analyze dependencies between components"""
        dependencies = {}
        
        for component in self.project_plan.components:
            # Look for import statements and dependency references in the component file
            deps = []
            try:
                with open(component.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find import statements
                import_lines = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        import_lines.append(line)
                
                # Extract module names from import statements
                for import_line in import_lines:
                    if import_line.startswith('import '):
                        modules = import_line[7:].split(',')
                        for module in modules:
                            module = module.strip().split('.')[0]  # Get base module name
                            if module and module not in deps:
                                deps.append(module)
                    elif import_line.startswith('from '):
                        parts = import_line.split()
                        if len(parts) > 1:
                            module = parts[1].split('.')[0]  # Get base module name
                            if module and module not in deps:
                                deps.append(module)
            
            except FileNotFoundError:
                print(f"Warning: Component file not found at {component.file_path}")
            
            dependencies[component.name] = deps
        
        return dependencies
    
    def generate_integration_layer(self, adapter_file_path: str = "src/integration_adapter.py") -> str:
        """Generate an integration layer that connects all components"""
        # Analyze component dependencies
        deps = self.analyze_component_dependencies()
        
        # Create import statements for all components
        imports = set()
        for component in self.project_plan.components:
            # Add import for each component
            module_path = component.file_path.replace('.py', '').replace('/', '.').replace('\\', '.')
            if component.component_type in ['function', 'class']:
                component_name = component.name
                imports.add(f"from {module_path} import {component_name}")
        
        # Generate the integration layer
        system_prompt = (
            "You are a senior Python architect. Generate an integration layer that connects "
            "the provided components. The integration layer should handle dependencies between "
            "components, manage data flow between them, and provide a unified interface. "
            "Follow common integration patterns and best practices."
        )
        
        components_info = []
        for component in self.project_plan.components:
            components_info.append(
                f"- {component.component_type} '{component.name}' in {component.file_path}: {component.description}"
            )
        
        user_prompt = (
            f"Component imports:\n" + "\n".join(imports) + 
            f"\n\nComponents to integrate:\n" + "\n".join(components_info) +
            f"\n\nGenerate the integration layer:"
        )
        
        llm = self.ai_assistant._get_llm_instance()
        integration_code = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1024
        )
        
        # Write the integration layer to file
        Path(adapter_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(adapter_file_path, 'w', encoding='utf-8') as f:
            f.write("# Auto-generated integration layer\n")
            f.write("# This file integrates all project components\n\n")
            f.write("\n".join(imports))
            f.write("\n\n")
            f.write(integration_code)
        
        print(f"Generated integration layer at {adapter_file_path}")
        
        return integration_code
    
    def validate_integration(self, adapter_file_path: str = "src/integration_adapter.py") -> Dict[str, str]:
        """Validate that all components can be integrated properly"""
        validation_results = {
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        try:
            # Check if the integration file exists
            if not Path(adapter_file_path).exists():
                validation_results["valid"] = False
                validation_results["issues"].append(f"Integration file does not exist: {adapter_file_path}")
                return validation_results
            
            # Try to compile the integration file
            with open(adapter_file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            compile(code, adapter_file_path, 'exec')
            
            # Try to import the integration module
            spec = importlib.util.spec_from_file_location("integration_adapter", adapter_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            print(f"[OK] Integration validation successful for {adapter_file_path}")
            
        except SyntaxError as e:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Syntax error in {adapter_file_path}: {str(e)}")
        except ImportError as e:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Import error in {adapter_file_path}: {str(e)}")
        except Exception as e:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Error validating {adapter_file_path}: {str(e)}")
        
        # Additional checks for component integration
        for component in self.project_plan.components:
            try:
                # Try to import each component
                module_path = component.file_path.replace('.py', '').replace('/', '.').replace('\\', '.')
                if component.component_type in ['function', 'class']:
                    __import__(module_path)
            except ImportError as e:
                validation_results["warnings"].append(f"Could not import component {component.name}: {str(e)}")
            except Exception as e:
                validation_results["warnings"].append(f"Error importing component {component.name}: {str(e)}")
        
        return validation_results
    
    def create_integration_test(self, test_file_path: str = "tests/test_integration.py") -> str:
        """Create integration tests for the connected components"""
        # Get information about components for test generation
        components_info = []
        for component in self.project_plan.components:
            components_info.append(
                f"- {component.component_type} '{component.name}' in {component.file_path}: {component.description}"
            )
        
        # Generate integration tests
        system_prompt = (
            "You are a senior Python test engineer. Generate comprehensive integration tests that "
            "test the interaction between multiple components. Focus on data flow, component "
            "communication, and end-to-end functionality. The tests should cover the integration "
            "layer and verify that components work together correctly."
        )
        
        user_prompt = (
            f"Components to test for integration:\n" + "\n".join(components_info) +
            f"\n\nGenerate integration tests:"
        )
        
        llm = self.ai_assistant._get_llm_instance()
        integration_tests = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1024
        )
        
        # Write the integration tests to file
        Path(test_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write("# Auto-generated integration tests\n")
            f.write("# These tests verify component integration\n\n")
            f.write("import pytest\n\n")
            f.write(integration_tests)
        
        print(f"Generated integration tests at {test_file_path}")
        
        return integration_tests
    
    def run_integration_tests(self, test_file_path: str = "tests/test_integration.py") -> Dict[str, any]:
        """Run the integration tests and return results"""
        try:
            # Check if test file exists
            if not Path(test_file_path).exists():
                return {
                    "return_code": -1,
                    "error": f"Integration test file does not exist: {test_file_path}"
                }
            
            # Run the integration tests
            cmd = [sys.executable, "-m", "pytest", test_file_path, "-v"]
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
    
    def create_api_layer(self, api_file_path: str = "src/api.py") -> str:
        """Create an API layer that exposes the integrated components"""
        # Get information about components for API generation
        components_info = []
        for component in self.project_plan.components:
            components_info.append(
                f"- {component.component_type} '{component.name}' in {component.file_path}: {component.description}"
            )
        
        # Generate API layer
        system_prompt = (
            "You are a senior Python developer. Generate a REST API layer using Flask or FastAPI "
            "that exposes the provided components as endpoints. The API should follow REST principles "
            "and include proper request/response handling, error handling, and documentation. "
            "Consider the data flow between components and design appropriate endpoints."
        )
        
        user_prompt = (
            f"Components to expose via API:\n" + "\n".join(components_info) +
            f"\n\nGenerate the API layer:"
        )
        
        llm = self.ai_assistant._get_llm_instance()
        api_code = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1024
        )
        
        # Write the API layer to file
        Path(api_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(api_file_path, 'w', encoding='utf-8') as f:
            f.write("# Auto-generated API layer\n")
            f.write("# This API exposes integrated components\n\n")
            f.write(api_code)
        
        print(f"Generated API layer at {api_file_path}")
        
        return api_code
    
    def create_documentation(self, doc_file_path: str = "docs/INTEGRATION.md") -> str:
        """Create documentation for the integrated system"""
        # Create a description of all components
        components_description = []
        for component in self.project_plan.components:
            components_description.append(
                f"### {component.component_type} '{component.name}'\n"
                f"**File:** {component.file_path}\n"
                f"**Description:** {component.description}\n"
            )
        
        # Generate documentation
        system_prompt = (
            "You are a technical writer. Create comprehensive documentation for the integrated system. "
            "Document the architecture, component interactions, API endpoints, data flow, and usage examples. "
            "Include diagrams where appropriate and make it clear for other developers to understand and use the system."
        )
        
        user_prompt = (
            f"Project: {self.project_plan.name}\n\n"
            f"Project Description: {self.project_plan.description}\n\n"
            f"Components:\n" + "\n".join(components_description) +
            f"\n\nCreate comprehensive documentation:"
        )
        
        llm = self.ai_assistant._get_llm_instance()
        documentation = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=2048
        )
        
        # Write the documentation to file
        Path(doc_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(doc_file_path, 'w', encoding='utf-8') as f:
            f.write(f"# {self.project_plan.name} - Integration Documentation\n\n")
            f.write(documentation)
        
        print(f"Generated integration documentation at {doc_file_path}")
        
        return documentation