"""
Task Management System for semindex AI CLI
Provides todo list functionality and progress tracking for complex projects
"""
import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from .project_planner import ProjectPlan, ProjectPhase, ProjectTask


class TaskStatus(Enum):
    """Status of a task in the project"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class TaskProgress:
    """Tracks progress of a specific task"""
    task_name: str
    status: TaskStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    notes: str = ""
    percentage_complete: int = 0  # 0-100


@dataclass
class PhaseProgress:
    """Tracks progress of a project phase"""
    phase_name: str
    status: TaskStatus
    tasks: List[TaskProgress]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    notes: str = ""


@dataclass
class ProjectProgress:
    """Tracks overall progress of a project"""
    project_name: str
    status: TaskStatus
    phases: List[PhaseProgress]
    created_at: datetime
    updated_at: datetime
    notes: str = ""


class TaskManager:
    """Manages task lists and progress tracking for AI projects"""
    
    def __init__(self, project_plan: ProjectPlan, storage_dir: str = ".semindex/tasks"):
        self.project_plan = project_plan
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress tracker
        self.progress = self._initialize_progress()
    
    def _initialize_progress(self) -> ProjectProgress:
        """Initialize progress tracking based on the project plan"""
        phases_progress = []
        
        for phase in self.project_plan.phases:
            tasks_progress = []
            for task in phase.tasks:
                tasks_progress.append(TaskProgress(
                    task_name=task.name,
                    status=TaskStatus.PENDING,
                    percentage_complete=0
                ))
            
            phases_progress.append(PhaseProgress(
                phase_name=phase.name,
                status=TaskStatus.PENDING,
                tasks=tasks_progress
            ))
        
        now = datetime.now()
        return ProjectProgress(
            project_name=self.project_plan.name,
            status=TaskStatus.PENDING,
            phases=phases_progress,
            created_at=now,
            updated_at=now
        )
    
    def mark_task_status(self, task_name: str, status: TaskStatus, notes: str = "") -> bool:
        """Update the status of a specific task"""
        for phase_progress in self.progress.phases:
            for task_progress in phase_progress.tasks:
                if task_progress.task_name == task_name:
                    task_progress.status = status
                    task_progress.notes = notes
                    task_progress.updated_at = datetime.now()
                    
                    # Update start/end times based on status
                    if status == TaskStatus.IN_PROGRESS and not task_progress.start_time:
                        task_progress.start_time = datetime.now()
                    elif status == TaskStatus.COMPLETED and not task_progress.end_time:
                        task_progress.end_time = datetime.now()
                    
                    # Update phase status based on tasks
                    self._update_phase_status(phase_progress)
                    
                    # Update project status based on phases
                    self._update_project_status()
                    
                    return True
        
        return False
    
    def mark_phase_status(self, phase_name: str, status: TaskStatus, notes: str = "") -> bool:
        """Update the status of a specific phase"""
        for phase_progress in self.progress.phases:
            if phase_progress.phase_name == phase_name:
                phase_progress.status = status
                phase_progress.notes = notes
                phase_progress.updated_at = datetime.now()
                
                # Update start/end times based on status
                if status == TaskStatus.IN_PROGRESS and not phase_progress.start_time:
                    phase_progress.start_time = datetime.now()
                elif status == TaskStatus.COMPLETED and not phase_progress.end_time:
                    phase_progress.end_time = datetime.now()
                
                # Update project status based on phases
                self._update_project_status()
                
                return True
        
        return False
    
    def _update_phase_status(self, phase_progress: PhaseProgress) -> None:
        """Update the status of a phase based on its tasks"""
        task_statuses = [task.status for task in phase_progress.tasks]
        
        if all(status == TaskStatus.COMPLETED for status in task_statuses):
            phase_progress.status = TaskStatus.COMPLETED
            if not phase_progress.end_time:
                phase_progress.end_time = datetime.now()
        elif any(status == TaskStatus.IN_PROGRESS for status in task_statuses):
            phase_progress.status = TaskStatus.IN_PROGRESS
        elif any(status == TaskStatus.BLOCKED for status in task_statuses):
            phase_progress.status = TaskStatus.BLOCKED
        elif all(status in [TaskStatus.CANCELLED, TaskStatus.COMPLETED] for status in task_statuses):
            phase_progress.status = TaskStatus.CANCELLED
        else:
            phase_progress.status = TaskStatus.PENDING
    
    def _update_project_status(self) -> None:
        """Update the status of the overall project based on its phases"""
        phase_statuses = [phase.status for phase in self.progress.phases]
        
        if all(status == TaskStatus.COMPLETED for status in phase_statuses):
            self.progress.status = TaskStatus.COMPLETED
        elif any(status == TaskStatus.IN_PROGRESS for status in phase_statuses):
            self.progress.status = TaskStatus.IN_PROGRESS
        elif any(status == TaskStatus.BLOCKED for status in phase_statuses):
            self.progress.status = TaskStatus.BLOCKED
        elif all(status in [TaskStatus.CANCELLED, TaskStatus.COMPLETED] for status in phase_statuses):
            self.progress.status = TaskStatus.CANCELLED
        else:
            self.progress.status = TaskStatus.PENDING
        
        self.progress.updated_at = datetime.now()
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[TaskProgress]:
        """Get all tasks with a specific status"""
        tasks = []
        for phase_progress in self.progress.phases:
            for task_progress in phase_progress.tasks:
                if task_progress.status == status:
                    tasks.append(task_progress)
        return tasks
    
    def get_pending_tasks(self) -> List[TaskProgress]:
        """Get all pending tasks"""
        return self.get_tasks_by_status(TaskStatus.PENDING)
    
    def get_in_progress_tasks(self) -> List[TaskProgress]:
        """Get all in-progress tasks"""
        return self.get_tasks_by_status(TaskStatus.IN_PROGRESS)
    
    def get_completed_tasks(self) -> List[TaskProgress]:
        """Get all completed tasks"""
        return self.get_tasks_by_status(TaskStatus.COMPLETED)
    
    def get_blocked_tasks(self) -> List[TaskProgress]:
        """Get all blocked tasks"""
        return self.get_tasks_by_status(TaskStatus.BLOCKED)
    
    def get_next_task(self) -> Optional[TaskProgress]:
        """Get the next task that should be worked on"""
        # For now, return the first pending task
        # In the future, we could implement more sophisticated prioritization
        pending_tasks = self.get_pending_tasks()
        if pending_tasks:
            return pending_tasks[0]
        return None
    
    def add_note_to_task(self, task_name: str, note: str) -> bool:
        """Add a note to a specific task"""
        for phase_progress in self.progress.phases:
            for task_progress in phase_progress.tasks:
                if task_progress.task_name == task_name:
                    if task_progress.notes:
                        task_progress.notes += f"\n{note}"
                    else:
                        task_progress.notes = note
                    task_progress.updated_at = datetime.now()
                    return True
        return False
    
    def update_task_progress(self, task_name: str, percentage: int) -> bool:
        """Update the percentage completion of a task"""
        for phase_progress in self.progress.phases:
            for task_progress in phase_progress.tasks:
                if task_progress.task_name == task_name:
                    task_progress.percentage_complete = max(0, min(100, percentage))  # Clamp between 0-100
                    task_progress.updated_at = datetime.now()
                    
                    # If task is 100% complete, mark it as completed (if not already)
                    if percentage == 100 and task_progress.status != TaskStatus.COMPLETED:
                        task_progress.status = TaskStatus.COMPLETED
                        if not task_progress.end_time:
                            task_progress.end_time = datetime.now()
                        
                        # Update phase and project status as needed
                        self._update_phase_status(
                            next(p for p in self.progress.phases if task_name in [t.task_name for t in p.tasks])
                        )
                        self._update_project_status()
                    
                    return True
        return False
    
    def save_progress(self, filename: Optional[str] = None) -> None:
        """Save the current progress to a file"""
        if filename is None:
            filename = f"{self.project_plan.name.replace(' ', '_')}_progress.json"
        
        filepath = self.storage_dir / filename
        
        # Convert datetime objects to ISO format for JSON serialization
        progress_data = {
            "project_name": self.progress.project_name,
            "status": self.progress.status.value,
            "created_at": self.progress.created_at.isoformat(),
            "updated_at": self.progress.updated_at.isoformat(),
            "notes": self.progress.notes,
            "phases": [
                {
                    "phase_name": phase.phase_name,
                    "status": phase.status.value,
                    "start_time": phase.start_time.isoformat() if phase.start_time else None,
                    "end_time": phase.end_time.isoformat() if phase.end_time else None,
                    "notes": phase.notes,
                    "tasks": [
                        {
                            "task_name": task.task_name,
                            "status": task.status.value,
                            "start_time": task.start_time.isoformat() if task.start_time else None,
                            "end_time": task.end_time.isoformat() if task.end_time else None,
                            "notes": task.notes,
                            "percentage_complete": task.percentage_complete
                        }
                        for task in phase.tasks
                    ]
                }
                for phase in self.progress.phases
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2)
    
    def load_progress(self, filename: str) -> bool:
        """Load progress from a file"""
        filepath = self.storage_dir / filename
        if not filepath.exists():
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            # Convert the loaded data back to our objects
            self.progress = ProjectProgress(
                project_name=progress_data["project_name"],
                status=TaskStatus(progress_data["status"]),
                phases=[],
                created_at=datetime.fromisoformat(progress_data["created_at"]),
                updated_at=datetime.fromisoformat(progress_data["updated_at"]),
                notes=progress_data.get("notes", "")
            )
            
            for phase_data in progress_data["phases"]:
                tasks = []
                for task_data in phase_data["tasks"]:
                    task = TaskProgress(
                        task_name=task_data["task_name"],
                        status=TaskStatus(task_data["status"]),
                        start_time=datetime.fromisoformat(task_data["start_time"]) if task_data["start_time"] else None,
                        end_time=datetime.fromisoformat(task_data["end_time"]) if task_data["end_time"] else None,
                        notes=task_data["notes"],
                        percentage_complete=task_data["percentage_complete"]
                    )
                    tasks.append(task)
                
                phase = PhaseProgress(
                    phase_name=phase_data["phase_name"],
                    status=TaskStatus(phase_data["status"]),
                    tasks=tasks,
                    start_time=datetime.fromisoformat(phase_data["start_time"]) if phase_data["start_time"] else None,
                    end_time=datetime.fromisoformat(phase_data["end_time"]) if phase_data["end_time"] else None,
                    notes=phase_data.get("notes", "")
                )
                self.progress.phases.append(phase)
            
            return True
        except Exception as e:
            print(f"Error loading progress from {filename}: {e}")
            return False
    
    def generate_report(self) -> str:
        """Generate a textual report of the project progress"""
        report_lines = [
            f"Project Progress Report: {self.progress.project_name}",
            "=" * 50,
            f"Overall Status: {self.progress.status.value.title()}",
            f"Created: {self.progress.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Last Updated: {self.progress.updated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Add phase-by-phase breakdown
        for phase_progress in self.progress.phases:
            report_lines.append(f"Phase: {phase_progress.phase_name} [{phase_progress.status.value.upper()}]")
            report_lines.append(f"  Tasks: {len(phase_progress.tasks)} total")
            
            completed_tasks = [t for t in phase_progress.tasks if t.status == TaskStatus.COMPLETED]
            in_progress_tasks = [t for t in phase_progress.tasks if t.status == TaskStatus.IN_PROGRESS]
            pending_tasks = [t for t in phase_progress.tasks if t.status == TaskStatus.PENDING]
            blocked_tasks = [t for t in phase_progress.tasks if t.status == TaskStatus.BLOCKED]
            
            report_lines.append(f"    Completed: {len(completed_tasks)}")
            report_lines.append(f"    In Progress: {len(in_progress_tasks)}")
            report_lines.append(f"    Pending: {len(pending_tasks)}")
            report_lines.append(f"    Blocked: {len(blocked_tasks)}")
            
            # Add task details for non-completed tasks
            for task_progress in phase_progress.tasks:
                if task_progress.status != TaskStatus.COMPLETED:
                    report_lines.append(f"    - {task_progress.task_name}: {task_progress.status.value} ({task_progress.percentage_complete}%)")
                    if task_progress.notes:
                        report_lines.append(f"      Notes: {task_progress.notes}")
            
            report_lines.append("")
        
        # Add overall summary
        total_tasks = sum(len(phase.tasks) for phase in self.progress.phases)
        completed_tasks = sum(len([t for t in phase.tasks if t.status == TaskStatus.COMPLETED]) for phase in self.progress.phases)
        progress_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        report_lines.append(f"Overall Progress: {completed_tasks}/{total_tasks} tasks completed ({progress_percentage:.1f}%)")
        
        return "\n".join(report_lines)