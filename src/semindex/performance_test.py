"""
Performance test for semindex parallel indexing functionality
"""
import os
import tempfile
import time
from pathlib import Path

from .indexer import Indexer


def create_test_project():
    """Create a simple test project for performance testing"""
    # Create a temporary directory for the test project
    test_dir = tempfile.mkdtemp(prefix="semindex_test_")
    
    # Create some test Python files
    test_files = {
        "main.py": '''
"""Main application module"""
import math
import sys
from utils import calculate_distance, format_output
from models import Point

def main():
    """Main entry point"""
    print("Starting application...")
    
    # Create some points
    point1 = Point(0, 0)
    point2 = Point(3, 4)
    
    # Calculate distance
    distance = calculate_distance(point1, point2)
    
    # Format and print output
    result = format_output(f"Distance: {distance}")
    print(result)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
''',
        "utils.py": '''
"""Utility functions"""
import math
from typing import Tuple
from models import Point

def calculate_distance(point1: Point, point2: Point) -> float:
    """Calculate Euclidean distance between two points"""
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    return math.sqrt(dx * dx + dy * dy)

def format_output(text: str) -> str:
    """Format output text with timestamp"""
    import datetime
    timestamp = datetime.datetime.now().isoformat()
    return f"[{timestamp}] {text}"

def normalize_angle(angle: float) -> float:
    """Normalize angle to [0, 2π] range"""
    while angle < 0:
        angle += 2 * math.pi
    while angle >= 2 * math.pi:
        angle -= 2 * math.pi
    return angle
''',
        "models.py": '''
"""Data models"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class Point:
    """2D point representation"""
    x: float
    y: float
    
    def distance_to(self, other: "Point") -> float:
        """Calculate distance to another point"""
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx * dx + dy * dy) ** 0.5

@dataclass
class Rectangle:
    """Rectangle representation"""
    top_left: Point
    bottom_right: Point
    
    @property
    def width(self) -> float:
        """Get rectangle width"""
        return abs(self.bottom_right.x - self.top_left.x)
    
    @property
    def height(self) -> float:
        """Get rectangle height"""
        return abs(self.top_left.y - self.bottom_right.y)
    
    def area(self) -> float:
        """Calculate rectangle area"""
        return self.width * self.height
'''
    }
    
    # Write the test files
    for filename, content in test_files.items():
        filepath = os.path.join(test_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
    
    return test_dir


def test_parallel_indexing():
    """Test parallel indexing performance"""
    # Create test project
    test_dir = create_test_project()
    
    # Create indexer instance
    indexer = Indexer(index_dir=os.path.join(test_dir, ".semindex"))
    
    # Test sequential indexing
    start_time = time.time()
    indexer.index_path(test_dir, verbose=True)
    sequential_time = time.time() - start_time
    
    # Test parallel indexing
    start_time = time.time()
    indexer.index_path_parallel(test_dir, verbose=True)
    parallel_time = time.time() - start_time
    
    # Print results
    print(f"Sequential indexing time: {sequential_time:.2f} seconds")
    print(f"Parallel indexing time: {parallel_time:.2f} seconds")
    print(f"Speedup: {sequential_time / parallel_time:.2f}x" if parallel_time > 0 else "N/A")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    
    return sequential_time, parallel_time


if __name__ == "__main__":
    test_parallel_indexing()