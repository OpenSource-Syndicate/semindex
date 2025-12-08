"""
Component Generator for semindex
Automatically divides project phases into functions and classes
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .ai_implementation_assistant import AIImplementationAssistant
from .project_planner import ProjectPlan, ProjectComponent
from .model import Symbol


class ComponentGenerator:
    """Generates functions and classes based on project plan components"""
    
    def __init__(self, index_dir: str, project_plan: ProjectPlan):
        self.index_dir = index_dir
        self.project_plan = project_plan
        self.ai_assistant = AIImplementationAssistant(index_dir=index_dir, config_path=None)
        
    def generate_components(self) -> List[ProjectComponent]:
        """Generate all components defined in the project plan"""
        generated_components = []
        
        for component in self.project_plan.components:
            if component.component_type == "function":
                generated_code = self._generate_function(component)
            elif component.component_type == "class":
                generated_code = self._generate_class(component)
            elif component.component_type == "module":
                generated_code = self._generate_module(component)
            else:
                # For other types, generate a placeholder
                generated_code = self._generate_placeholder(component)
            
            # Write the generated code to the appropriate file
            self._write_component_code(component.file_path, generated_code, component)
            
            generated_components.append(component)
        
        return generated_components
    
    def _generate_function(self, component: ProjectComponent) -> str:
        """Generate a function based on the component specification"""
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
        """Generate a class based on the component specification"""
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
    
    def _generate_module(self, component: ProjectComponent) -> str:
        """Generate a module based on the component specification"""
        # For modules, we'll create a skeletal structure with appropriate imports and placeholders
        system_prompt = (
            "You are a senior Python developer. Generate a Python module structure based on "
            "the provided description. Include appropriate imports, constants, and function/class "
            "placeholders that would be typical for this kind of module. Follow common Python "
            "patterns and conventions."
        )
        
        user_prompt = f"Module description:\n{component.description}\n\nGenerate the module structure:"
        
        # Use a direct LLM call since modules are more structural
        llm = self.ai_assistant._get_llm_instance()
        module_code = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1024
        )
        
        return module_code
    
    def _generate_placeholder(self, component: ProjectComponent) -> str:
        """Generate a placeholder for unknown component types"""
        return f"# Placeholder for {component.component_type} '{component.name}'\n# Description: {component.description}\n\n# TODO: Implement this component\npass\n"
    
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
        else:
            # If the file doesn't exist, create it with the component
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Generated component: {component.name}\n")
                f.write(f"# Description: {component.description}\n\n")
                f.write(code)
    
    def _analyze_component_dependencies(self, component: ProjectComponent) -> List[str]:
        """Analyze and identify dependencies for a component"""
        # Use the AI to identify what imports/dependencies might be needed
        system_prompt = (
            "You are a senior Python developer. Analyze the provided component description "
            "and identify what Python modules or packages would need to be imported "
            "to implement this component. Return a list of import statements that would be needed."
        )
        
        user_prompt = f"Component: {component.name}\nDescription: {component.description}\n\nIdentify required imports:"
        
        llm = self.ai_assistant._get_llm_instance()
        response = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=256
        )
        
        # Extract import statements from the response
        imports = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
        
        return imports

    def divide_phase_into_components(self, phase_name: str) -> List[ProjectComponent]:
        """Divide a specific project phase into individual components (functions and classes)"""
        # Find the phase in the project plan
        target_phase = None
        for phase in self.project_plan.phases:
            if phase.name == phase_name:
                target_phase = phase
                break
        
        if not target_phase:
            raise ValueError(f"Phase '{phase_name}' not found in project plan")
        
        components = []
        
        # For each task in the phase, identify what components need to be created
        for task in target_phase.tasks:
            # Use AI to determine what functions/classes are needed for this task
            system_prompt = (
                "You are a senior software architect. Based on the provided task description, "
                "identify what functions and classes would need to be created to implement this task. "
                "Return the components in the following JSON format:\n"
                "[\n"
                "  {\n"
                "    \"name\": \"Component Name\",\n"
                "    \"description\": \"What the component does\",\n"
                "    \"component_type\": \"function|class|module\"\n"
                "  }\n"
                "]"
            )
            
            user_prompt = f"Task: {task.name}\nDescription: {task.description}\n\nIdentify the components needed:"
            
            llm = self.ai_assistant._get_llm_instance()
            response = llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=512
            )
            
            try:
                import json
                component_list = json.loads(response)
                
                for comp_data in component_list:
                    # Determine appropriate file path based on project structure
                    file_path = self._determine_file_path(comp_data["name"], comp_data["component_type"])
                    
                    component = ProjectComponent(
                        name=comp_data["name"],
                        description=comp_data["description"],
                        component_type=comp_data["component_type"],
                        file_path=file_path,
                        dependencies=comp_data.get("dependencies", [])
                    )
                    
                    components.append(component)
            except json.JSONDecodeError:
                # If response isn't valid JSON, create a basic component
                components.append(ProjectComponent(
                    name=f"{task.name.replace(' ', '_')}_implementation",
                    description=f"Implementation for task: {task.description}",
                    component_type="function",  # Default to function
                    file_path=self._determine_file_path(task.name, "function"),
                    dependencies=[]
                ))
        
        return components
    
    def _determine_file_path(self, component_name: str, component_type: str) -> str:
        """Determine an appropriate file path for a component based on the project structure"""
        # Convert component name to a valid file/module name
        module_name = re.sub(r'[^a-zA-Z0-9_]', '_', component_name.lower())
        
        # Determine appropriate location based on component type
        if component_type == "function":
            return f"src/{module_name}.py"
        elif component_type == "class":
            return f"src/{module_name}.py"
        elif component_type == "module":
            return f"src/{module_name}.py"
        else:
            return f"src/{module_name}.py"
    
    def _llm_generate(self, system_prompt: str, user_prompt: str) -> str:
        """Helper to generate content using the LLM - used by internal methods"""
        llm = self.ai_assistant._get_llm_instance()
        return llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1024
        )

    def _get_llm_instance(self):
        """Helper to get the LLM instance - used by internal methods"""
        from .local_llm import LocalLLM
        return LocalLLM(
            model_type="transformer",
            model_name=os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        )