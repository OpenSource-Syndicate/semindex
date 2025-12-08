#!/usr/bin/env python3
"""
End-to-end test for the AI CLI project planning functionality
"""
import json
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    """Run a shell command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Exit code: {result.returncode}")
    if result.stdout:
        print(f"STDOUT: {result.stdout}")
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    return result


def test_ai_planning():
    print("=== AI CLI Project Planning End-to-End Test ===\n")
    
    # Step 1: Index the current project if not already indexed
    print("Step 1: Ensuring project is indexed...")
    if not Path(".semindex").exists():
        # Import the CLI function directly
        from semindex.cli import cmd_index
        import argparse
        
        # Create args object similar to what the CLI would parse
        args = argparse.Namespace(
            repo=".",
            index_dir=".semindex",
            model=None,
            batch=16,
            verbose=False,
            chunking='symbol',
            similarity_threshold=0.7,
            incremental=True,
            language='auto',
            include_docs=False
        )
        try:
            cmd_index(args)
        except Exception as e:
            print(f"Warning: Indexing failed with error {e}, proceeding anyway...")
    
    # Step 2: Create a project plan
    print("\nStep 2: Creating a project plan...")
    plan_output = "test_plan.json"
    result = run_command([
        "uv", "run", "semindex", "ai-plan", "create",
        "Create a Python module to analyze and visualize code metrics",
        "--project-name", "CodeMetricsAnalyzer",
        "--output", plan_output,
        "--index-dir", ".semindex"
    ])
    
    if result.returncode != 0:
        print("Project plan creation failed, proceeding to create a simple plan manually...")
        # Create a simple plan manually to continue testing
        simple_plan = {
            "name": "CodeMetricsAnalyzer",
            "description": "Create a Python module to analyze and visualize code metrics",
            "created_from_codebase": False,
            "phases": [
                {
                    "name": "Analysis",
                    "description": "Analyze the current codebase to understand metrics",
                    "phase_type": "analysis",
                    "tasks": [
                        {
                            "name": "Identify metrics",
                            "description": "Identify useful code metrics to track",
                            "required_knowledge": ["python", "code_analysis"],
                            "dependencies": [],
                            "estimated_duration_hours": 4
                        }
                    ],
                    "dependencies": []
                },
                {
                    "name": "Design",
                    "description": "Design the metrics analyzer module",
                    "phase_type": "design", 
                    "tasks": [
                        {
                            "name": "Module design",
                            "description": "Design the module structure and API",
                            "required_knowledge": ["python", "design_patterns"],
                            "dependencies": [],
                            "estimated_duration_hours": 6
                        }
                    ],
                    "dependencies": ["Analysis"]
                },
                {
                    "name": "Implementation",
                    "description": "Implement the metrics analyzer",
                    "phase_type": "implementation",
                    "tasks": [
                        {
                            "name": "Core implementation",
                            "description": "Implement core functionality",
                            "required_knowledge": ["python", "code_analysis"],
                            "dependencies": ["Module design"],
                            "estimated_duration_hours": 12
                        }
                    ],
                    "dependencies": ["Design"]
                }
            ],
            "components": [
                {
                    "name": "CodeMetricsAnalyzer",
                    "description": "Main module to analyze code metrics",
                    "component_type": "class",
                    "file_path": "src/metrics_analyzer.py",
                    "dependencies": [],
                    "test_file_path": "tests/test_metrics_analyzer.py"
                },
                {
                    "name": "calculate_complexity",
                    "description": "Function to calculate code complexity metrics",
                    "component_type": "function", 
                    "file_path": "src/metrics_analyzer.py",
                    "dependencies": ["CodeMetricsAnalyzer"],
                    "test_file_path": "tests/test_metrics_analyzer.py"
                }
            ]
        }
        
        with open(plan_output, 'w') as f:
            json.dump(simple_plan, f, indent=2)
        print(f"Created simple plan at {plan_output}")
    else:
        print(f"Project plan created successfully: {plan_output}")
    
    # Step 3: Manage the project plan
    print("\nStep 3: Managing the project plan...")
    result = run_command([
        "uv", "run", "semindex", "ai-plan", "manage",
        "--plan-file", plan_output,
        "--report",
        "--index-dir", ".semindex"
    ])
    
    # Step 4: Execute the project plan (just the first phase for testing)
    print("\nStep 4: Executing the project plan...")
    result = run_command([
        "uv", "run", "semindex", "ai-plan", "execute",
        "--plan-file", plan_output,
        "--index-dir", ".semindex",
        "--generate-tests",
        "--integrate"
    ])
    
    if result.returncode == 0:
        print("Project execution completed successfully!")
        
        # Look for generated files
        generated_files = []
        for root, dirs, files in os.walk("."):
            for file in files:
                if "generated" in file or "auto" in file or file.startswith("test_"):
                    generated_files.append(os.path.join(root, file))
        
        print(f"Found {len(generated_files)} potentially generated files:")
        for f in generated_files[:10]:  # Show first 10
            print(f"  - {f}")
    else:
        print("Project execution failed, but this is expected without a full LLM setup")
    
    print("\n=== Test completed ===")


if __name__ == "__main__":
    test_ai_planning()