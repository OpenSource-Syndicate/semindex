"""
Development Workflow for semindex AI CLI
Manages the workflow of creating functions and classes based on project plans
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .component_generator import ComponentGenerator
from .project_planner import ProjectPlan, ProjectComponent
from .task_manager import TaskManager, TaskStatus
from .ai_implementation_assistant import AIImplementationAssistant


class DevelopmentWorkflow:
    """Manages the development workflow for AI-generated projects"""
    
    def __init__(self, project_plan: ProjectPlan, task_manager: TaskManager, index_dir: str):
        self.project_plan = project_plan
        self.task_manager = task_manager
        self.index_dir = index_dir
        self.component_generator = ComponentGenerator(index_dir, project_plan)
        self.ai_assistant = AIImplementationAssistant(index_dir, config_path=None)
        
        # Create a directory for development artifacts
        self.dev_dir = Path(".semindex/dev")
        self.dev_dir.mkdir(parents=True, exist_ok=True)
    
    def execute_phase(self, phase_name: str) -> bool:
        """Execute all tasks for a specific phase"""
        # Find the phase in the plan
        target_phase = None
        for phase in self.project_plan.phases:
            if phase.name == phase_name:
                target_phase = phase
                break
        
        if not target_phase:
            print(f"Phase '{phase_name}' not found in project plan")
            return False
        
        print(f"Starting execution of phase: {phase_name}")
        
        # Mark phase as in progress
        self.task_manager.mark_phase_status(phase_name, TaskStatus.IN_PROGRESS)
        
        try:
            # Generate components for this phase
            components = self.component_generator.divide_phase_into_components(phase_name)
            
            # Add these components to the project plan
            self.project_plan.components.extend(components)
            
            # Generate code for each component
            for component in components:
                print(f"Generating component: {component.name}")
                
                # Generate the component code
                if component.component_type == "function":
                    code = self._generate_function(component)
                elif component.component_type == "class":
                    code = self._generate_class(component)
                else:
                    code = self._generate_placeholder(component)
                
                # Write the component to file
                self._write_component_code(component.file_path, code, component)
                
                # Update task status
                task_name = f"Create {component.name}"
                self.task_manager.mark_task_status(task_name, TaskStatus.COMPLETED, 
                                                 f"Generated {component.component_type} {component.name}")
            
            # After all components are generated, update phase status
            self.task_manager.mark_phase_status(phase_name, TaskStatus.COMPLETED, 
                                              f"Completed all component generation for {phase_name}")
            
            print(f"Completed execution of phase: {phase_name}")
            return True
            
        except Exception as e:
            print(f"Error executing phase '{phase_name}': {str(e)}")
            self.task_manager.mark_phase_status(phase_name, TaskStatus.BLOCKED, 
                                              f"Error during execution: {str(e)}")
            return False
    
    def execute_project(self) -> bool:
        """Execute the entire project following the plan"""
        print("Starting execution of the entire project")
        
        success = True
        
        # Execute each phase in order
        for phase in self.project_plan.phases:
            if not self.execute_phase(phase.name):
                success = False
                print(f"Failed to execute phase: {phase.name}")
        
        if success:
            print("Project execution completed successfully")
        else:
            print("Project execution completed with some errors")
        
        return success
    
    def _generate_function(self, component: ProjectComponent) -> str:
        """Generate a function component"""
        # Get context from similar functions in the codebase
        context = self.ai_assistant.generate_function_signature(component.description)
        
        # Generate the full function implementation
        system_prompt = (
            "You are a senior Python developer. Generate a complete function implementation "
            "based on the provided signature and description. Follow the code patterns and "
            "conventions from the context provided. Include appropriate error handling, "
            "documentation, and type hints where applicable."
        )
        
        user_prompt = f"Function signature:\n{context}\n\nFunction description:\n{component.description}\n\nGenerate the complete implementation:"
        
        llm = self.ai_assistant._get_llm_instance()
        function_code = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1024
        )
        
        return function_code
    
    def _generate_class(self, component: ProjectComponent) -> str:
        """Generate a class component"""
        # Get context about similar classes in the codebase
        context = self.ai_assistant.generate_class_structure(component.description)
        
        # Generate the full class implementation
        system_prompt = (
            "You are a senior Python developer. Generate a complete class implementation "
            "based on the provided structure and description. Follow the code patterns and "
            "conventions from the context provided. Include appropriate methods, attributes, "
            "documentation, and follow object-oriented design principles."
        )
        
        user_prompt = f"Class structure:\n{context}\n\nClass description:\n{component.description}\n\nGenerate the complete implementation:"
        
        llm = self.ai_assistant._get_llm_instance()
        class_code = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1024
        )
        
        return class_code
    
    def _generate_placeholder(self, component: ProjectComponent) -> str:
        """Generate a placeholder component"""
        return f"# Generated component: {component.name}\n# Description: {component.description}\n\n# TODO: Implement this component\npass\n"
    
    def _write_component_code(self, file_path: str, code: str, component: ProjectComponent) -> None:
        """Write the generated component code to the appropriate file"""
        # Create directories if they don't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists
        if Path(file_path).exists():
            # If the file exists, append the new component to it
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
                
            # Append the new component code
            if code.strip() not in existing_content:
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n\n# Generated component: {component.name}\n")
                    f.write(f"# Description: {component.description}\n")
                    f.write(code)
                
                print(f"Appended component {component.name} to {file_path}")
        else:
            # If the file doesn't exist, create it with the component
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Generated component: {component.name}\n")
                f.write(f"# Description: {component.description}\n\n")
                f.write(code)
            
            print(f"Created new file {file_path} with component {component.name}")
    
    def validate_generated_code(self) -> List[str]:
        """Validate the syntax of all generated code files"""
        validation_errors = []
        
        for component in self.project_plan.components:
            try:
                file_path = Path(component.file_path)
                if file_path.exists():
                    # Read the file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    
                    # Validate syntax by attempting to compile
                    compile(code, str(file_path), 'exec')
                    print(f"[OK] Valid syntax in {file_path}")
                else:
                    validation_errors.append(f"File does not exist: {component.file_path}")
            except SyntaxError as e:
                validation_errors.append(f"Syntax error in {component.file_path}: {str(e)}")
            except Exception as e:
                validation_errors.append(f"Error validating {component.file_path}: {str(e)}")
        
        if validation_errors:
            print("\nValidation errors found:")
            for error in validation_errors:
                print(f"  - {error}")
        else:
            print("\n[OK] All generated code files have valid syntax")
        
        return validation_errors
    
    def run_code_analysis(self) -> str:
        """Run code analysis on the generated components"""
        analysis_results = []
        
        for component in self.project_plan.components:
            file_path = Path(component.file_path)
            if file_path.exists():
                analysis = self.ai_assistant.check_code_quality(str(file_path))
                analysis_results.append({
                    'file': component.file_path,
                    'component': component.name,
                    'analysis': analysis
                })
        
        # Generate a summary of the analysis
        summary = [f"Code Analysis for Project: {self.project_plan.name}"]
        summary.append("=" * 50)
        
        for result in analysis_results:
            summary.append(f"\nComponent: {result['component']} ({result['file']})")
            summary.append("-" * 30)
            summary.append(result['analysis'])
            summary.append("")
        
        return "\n".join(summary)
    
    def generate_task_report(self) -> str:
        """Generate a report of completed tasks"""
        return self.task_manager.generate_report()