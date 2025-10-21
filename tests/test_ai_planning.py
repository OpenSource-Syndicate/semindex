#!/usr/bin/env python3
"""
Test script to verify the AI CLI planning functionality
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semindex.project_planner import ProjectPlanner
from semindex.task_manager import TaskManager
from semindex.development_workflow import DevelopmentWorkflow


def test_project_planning():
    print("Testing Project Planning functionality...")
    
    # Initialize planner
    planner = ProjectPlanner(index_dir='.semindex')
    
    # Create a simple project plan
    project_description = 'Create a web API to manage tasks with CRUD operations'
    plan = planner.generate_plan_from_description(project_description, 'TaskAPI')
    
    print(f'Generated plan: {plan.name}')
    print(f'Description: {plan.description}')
    print(f'Phases: {len(plan.phases)}')
    print(f'Components: {len(plan.components)}')
    
    if plan.phases:
        for i, phase in enumerate(plan.phases[:3]):  # Print first 3 phases
            print(f'  Phase {i+1}: {phase.name} [{phase.phase_type.value}]')
            print(f'    Description: {phase.description}')
            print(f'    Tasks: {len(phase.tasks)}')
            if phase.tasks:
                print(f'    First task: {phase.tasks[0].name}')
    
    if plan.components:
        for i, component in enumerate(plan.components[:3]):  # Print first 3 components
            print(f'  Component {i+1}: {component.name} ({component.component_type}) in {component.file_path}')
    
    print("\nTesting Task Management functionality...")
    
    # Initialize task manager
    task_manager = TaskManager(project_plan=plan)
    
    # Show initial task status
    pending_tasks = task_manager.get_pending_tasks()
    print(f'Initial pending tasks: {len(pending_tasks)}')
    
    # Mark a task as in progress
    if pending_tasks:
        task_name = pending_tasks[0].task_name
        task_manager.mark_task_status(task_name, task_manager.TaskStatus.IN_PROGRESS)
        print(f'Marked task "{task_name}" as IN_PROGRESS')
        
        # Update progress
        task_manager.update_task_progress(task_name, 50)
        print(f'Updated progress for "{task_name}" to 50%')
    
    print("\nTesting Development Workflow...")
    
    # Initialize development workflow
    workflow = DevelopmentWorkflow(project_plan=plan, task_manager=task_manager, index_dir='.semindex')
    
    # Generate a report
    report = task_manager.generate_report()
    print(f"Generated task report with {len(task_manager.progress.phases)} phases")
    
    print("\nAll tests completed successfully!")
    return True


if __name__ == "__main__":
    test_project_planning()