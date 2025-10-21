"""
Project Planner Module for semindex
Provides AI-powered project planning capabilities for complex software development
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .ai import FunctionCallExecutor
from .model import Symbol


class PhaseType(Enum):
    """Types of project phases"""
    ANALYSIS = "analysis"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    DOCUMENTATION = "documentation"


@dataclass
class ProjectPhase:
    """Represents a phase in the project plan"""
    name: str
    description: str
    phase_type: PhaseType
    tasks: List['ProjectTask']
    dependencies: List[str]  # Names of phases this phase depends on
    estimated_duration_hours: int = 8


@dataclass
class ProjectTask:
    """Represents a specific task within a project phase"""
    name: str
    description: str
    required_knowledge: List[str]  # Skills or technologies needed
    dependencies: List[str]  # Names of tasks this task depends on
    estimated_duration_hours: int
    component_path: Optional[str] = None  # Path to the component being developed


@dataclass
class ProjectComponent:
    """Represents a software component to be developed"""
    name: str
    description: str
    component_type: str  # 'function', 'class', 'module', 'api_endpoint', etc.
    file_path: str
    dependencies: List[str]  # Other components this component depends on
    test_file_path: Optional[str] = None


@dataclass
class ProjectPlan:
    """Complete project plan with phases, tasks, and components"""
    name: str
    description: str
    phases: List[ProjectPhase]
    components: List[ProjectComponent]
    created_from_codebase: bool = False  # Whether this plan was generated from analyzing existing code


class ProjectPlanner:
    """Main class for creating AI-powered project plans"""
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.function_executor = FunctionCallExecutor(index_dir=index_dir)
        
    def generate_plan_from_description(self, project_description: str, project_name: str = "NewProject") -> ProjectPlan:
        """Generate a project plan from a natural language description"""
        # Use AI to analyze the description and create phases
        system_prompt = (
            "You are a software project planning expert. Based on the project description, "
            "create a comprehensive project plan with phases, tasks, and components. "
            "Return the plan in JSON format with the following structure:\n"
            "{\n"
            "  \"name\": \"Project Name\",\n"
            "  \"description\": \"Project Description\",\n"
            "  \"phases\": [\n"
            "    {\n"
            "      \"name\": \"Phase Name\",\n"
            "      \"description\": \"Phase Description\",\n"
            "      \"phase_type\": \"analysis|design|implementation|testing|deployment|documentation\",\n"
            "      \"tasks\": [\n"
            "        {\n"
            "          \"name\": \"Task Name\",\n"
            "          \"description\": \"Task Description\",\n"
            "          \"required_knowledge\": [\"technology1\", \"technology2\"],\n"
            "          \"estimated_duration_hours\": 4\n"
            "        }\n"
            "      ],\n"
            "      \"dependencies\": [\"previous_phase_name\"]\n"
            "    }\n"
            "  ],\n"
            "  \"components\": [\n"
            "    {\n"
            "      \"name\": \"Component Name\",\n"
            "      \"description\": \"Component Description\",\n"
            "      \"component_type\": \"function|class|module|api_endpoint\",\n"
            "      \"file_path\": \"path/to/file.py\",\n"
            "      \"dependencies\": [\"other_component_name\"]\n"
            "    }\n"
            "  ]\n"
            "}"
        )
        
        user_prompt = f"Create a project plan for: {project_description}"
        
        try:
            # For now, we'll use the local LLM directly
            from .local_llm import LocalLLM
            llm = LocalLLM(
                model_type="transformer",
                model_name=os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            )
            
            response = llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=2048
            )
            
            # Parse the JSON response
            try:
                plan_data = json.loads(response)
                return self._deserialize_plan(plan_data)
            except json.JSONDecodeError:
                # If the response isn't valid JSON, create a basic plan
                return ProjectPlan(
                    name=project_name,
                    description=project_description,
                    phases=[
                        ProjectPhase(
                            name="Initial Analysis",
                            description="Analyze the project requirements",
                            phase_type=PhaseType.ANALYSIS,
                            tasks=[
                                ProjectTask(
                                    name="Requirements Analysis",
                                    description=f"Analyze the project requirements based on: {project_description}",
                                    required_knowledge=["requirements_analysis"],
                                    dependencies=[],
                                    estimated_duration_hours=4
                                )
                            ],
                            dependencies=[]
                        ),
                        ProjectPhase(
                            name="Design",
                            description="Design the system architecture",
                            phase_type=PhaseType.DESIGN,
                            tasks=[
                                ProjectTask(
                                    name="System Design",
                                    description="Create system architecture",
                                    required_knowledge=["system_design"],
                                    dependencies=[],
                                    estimated_duration_hours=8
                                )
                            ],
                            dependencies=["Initial Analysis"]
                        ),
                        ProjectPhase(
                            name="Implementation",
                            description="Implement the main functionality",
                            phase_type=PhaseType.IMPLEMENTATION,
                            tasks=[
                                ProjectTask(
                                    name="Core Implementation",
                                    description="Implement the core features",
                                    required_knowledge=["programming"],
                                    dependencies=["System Design"],
                                    estimated_duration_hours=16
                                )
                            ],
                            dependencies=["Design"]
                        )
                    ],
                    components=[]
                )
        except Exception as e:
            print(f"Error generating plan from description: {e}")
            # Return a basic plan in case of error
            return ProjectPlan(
                name=project_name,
                description=project_description,
                phases=[],
                components=[]
            )
    
    def generate_plan_from_codebase(self, repo_root: str, project_description: str = "") -> ProjectPlan:
        """Generate a project plan by analyzing the existing codebase"""
        # Search the codebase for relevant information
        context_snippets = self.function_executor.search_code(query="main components architecture design patterns", top_k=10)
        
        # Build a prompt with codebase context
        context_text = "\n".join([f"File: {item.get('path', 'Unknown')}\n{item.get('snippet', '')[:500]}" for item in context_snippets if "error" not in item])
        
        system_prompt = (
            "You are a software project planning expert. Analyze the provided codebase context and project description "
            "to create a comprehensive project plan with phases, tasks, and components. "
            "Consider the existing architecture and suggest improvements or extensions. "
            "Return the plan in JSON format with the following structure:\n"
            "{\n"
            "  \"name\": \"Project Name\",\n"
            "  \"description\": \"Project Description\",\n"
            "  \"phases\": [\n"
            "    {\n"
            "      \"name\": \"Phase Name\",\n"
            "      \"description\": \"Phase Description\",\n"
            "      \"phase_type\": \"analysis|design|implementation|testing|deployment|documentation\",\n"
            "      \"tasks\": [\n"
            "        {\n"
            "          \"name\": \"Task Name\",\n"
            "          \"description\": \"Task Description\",\n"
            "          \"required_knowledge\": [\"technology1\", \"technology2\"],\n"
            "          \"estimated_duration_hours\": 4\n"
            "        }\n"
            "      ],\n"
            "      \"dependencies\": [\"previous_phase_name\"]\n"
            "    }\n"
            "  ],\n"
            "  \"components\": [\n"
            "    {\n"
            "      \"name\": \"Component Name\",\n"
            "      \"description\": \"Component Description\",\n"
            "      \"component_type\": \"function|class|module|api_endpoint\",\n"
            "      \"file_path\": \"path/to/file.py\",\n"
            "      \"dependencies\": [\"other_component_name\"]\n"
            "    }\n"
            "  ]\n"
            "}"
        )
        
        user_prompt = f"Project Description: {project_description}\n\nCodebase Context:\n{context_text}"
        
        try:
            from .local_llm import LocalLLM
            llm = LocalLLM(
                model_type="transformer",
                model_name=os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            )
            
            response = llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=2048
            )
            
            # Parse the JSON response
            try:
                plan_data = json.loads(response)
                plan = self._deserialize_plan(plan_data)
                plan.created_from_codebase = True
                return plan
            except json.JSONDecodeError:
                print("Could not parse AI response as JSON, creating basic plan from codebase analysis")
                # Create a simple plan based on the codebase
                components = self._extract_components_from_codebase(repo_root)
                return ProjectPlan(
                    name="Codebase Extension Plan",
                    description=f"Plan for extending the existing codebase: {project_description}",
                    phases=[
                        ProjectPhase(
                            name="Analysis",
                            description="Analyze the current codebase structure",
                            phase_type=PhaseType.ANALYSIS,
                            tasks=[
                                ProjectTask(
                                    name="Codebase Analysis",
                                    description="Analyze the current codebase to understand architecture",
                                    required_knowledge=["python", "existing_frameworks"],
                                    dependencies=[],
                                    estimated_duration_hours=8
                                )
                            ],
                            dependencies=[]
                        ),
                        ProjectPhase(
                            name="Design",
                            description="Design new components to extend functionality",
                            phase_type=PhaseType.DESIGN,
                            tasks=[
                                ProjectTask(
                                    name="Extension Design",
                                    description="Design new components integrating with the existing architecture",
                                    required_knowledge=["system_design"],
                                    dependencies=["Codebase Analysis"],
                                    estimated_duration_hours=8
                                )
                            ],
                            dependencies=["Analysis"]
                        ),
                        ProjectPhase(
                            name="Implementation",
                            description="Implement new components",
                            phase_type=PhaseType.IMPLEMENTATION,
                            tasks=[
                                ProjectTask(
                                    name="Component Implementation",
                                    description="Implement new components following the design",
                                    required_knowledge=["python", "existing_frameworks"],
                                    dependencies=["Extension Design"],
                                    estimated_duration_hours=24
                                )
                            ],
                            dependencies=["Design"]
                        )
                    ],
                    components=components
                )
        except Exception as e:
            print(f"Error generating plan from codebase: {e}")
            return ProjectPlan(
                name="Basic Plan",
                description="Basic plan due to error in codebase analysis",
                phases=[],
                components=[],
                created_from_codebase=True
            )
    
    def _extract_components_from_codebase(self, repo_root: str) -> List[ProjectComponent]:
        """Extract component information from the codebase"""
        components = []
        
        # Get all symbols from the index to identify potential components
        try:
            from .store import DB_NAME, db_conn
            db_path = os.path.join(self.index_dir, DB_NAME)
            
            with db_conn(db_path) as con:
                cur = con.cursor()
                cur.execute(
                    "SELECT path, name, kind, start_line, end_line, signature FROM symbols "
                    "WHERE kind IN ('function', 'class', 'module') ORDER BY path"
                )
                rows = cur.fetchall()
                
                for row in rows:
                    path, name, kind, start_line, end_line, signature = row
                    # Create a component for each symbol
                    component_type = kind
                    if kind == 'module':
                        # Module components are typically full files
                        file_path = path
                    else:
                        # Function/class components are within files
                        file_path = path
                    
                    components.append(ProjectComponent(
                        name=name,
                        description=f"{kind.capitalize()} {name} in {path}",
                        component_type=component_type,
                        file_path=file_path
                    ))
        except Exception as e:
            print(f"Error extracting components from codebase: {e}")
        
        return components
    
    def _deserialize_plan(self, plan_data: dict) -> ProjectPlan:
        """Convert a dictionary representation to a ProjectPlan object"""
        phases = []
        for phase_data in plan_data.get("phases", []):
            tasks = []
            for task_data in phase_data.get("tasks", []):
                tasks.append(ProjectTask(
                    name=task_data["name"],
                    description=task_data["description"],
                    required_knowledge=task_data.get("required_knowledge", []),
                    dependencies=task_data.get("dependencies", []),
                    estimated_duration_hours=task_data.get("estimated_duration_hours", 4)
                ))
            
            phases.append(ProjectPhase(
                name=phase_data["name"],
                description=phase_data["description"],
                phase_type=PhaseType(phase_data["phase_type"]),
                tasks=tasks,
                dependencies=phase_data.get("dependencies", [])
            ))
        
        components = []
        for component_data in plan_data.get("components", []):
            components.append(ProjectComponent(
                name=component_data["name"],
                description=component_data["description"],
                component_type=component_data["component_type"],
                file_path=component_data["file_path"],
                dependencies=component_data.get("dependencies", []),
                test_file_path=component_data.get("test_file_path")
            ))
        
        return ProjectPlan(
            name=plan_data["name"],
            description=plan_data["description"],
            phases=phases,
            components=components
        )
    
    def save_plan(self, plan: ProjectPlan, file_path: str) -> None:
        """Save a project plan to a JSON file"""
        plan_data = {
            "name": plan.name,
            "description": plan.description,
            "created_from_codebase": plan.created_from_codebase,
            "phases": [
                {
                    "name": phase.name,
                    "description": phase.description,
                    "phase_type": phase.phase_type.value,
                    "tasks": [
                        {
                            "name": task.name,
                            "description": task.description,
                            "required_knowledge": task.required_knowledge,
                            "dependencies": task.dependencies,
                            "estimated_duration_hours": task.estimated_duration_hours
                        }
                        for task in phase.tasks
                    ],
                    "dependencies": phase.dependencies
                }
                for phase in plan.phases
            ],
            "components": [
                {
                    "name": component.name,
                    "description": component.description,
                    "component_type": component.component_type,
                    "file_path": component.file_path,
                    "dependencies": component.dependencies,
                    "test_file_path": component.test_file_path
                }
                for component in plan.components
            ]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(plan_data, f, indent=2)
    
    def load_plan(self, file_path: str) -> ProjectPlan:
        """Load a project plan from a JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            plan_data = json.load(f)
        return self._deserialize_plan(plan_data)