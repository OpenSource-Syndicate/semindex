"""
Context Watcher Module for semindex
Provides real-time context updating as users modify code in the project.
"""
import os
import time
import threading
from pathlib import Path
from typing import Dict, Callable, Set, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .context_generator import ContextCache


class ContextFileWatcher(FileSystemEventHandler):
    """Watches file changes and updates context cache accordingly"""
    
    def __init__(self, index_dir: str, cache: ContextCache, 
                 on_change_callback: Optional[Callable] = None):
        super().__init__()
        self.index_dir = index_dir
        self.cache = cache
        self.on_change_callback = on_change_callback
        self.observer = Observer()
        self.watched_files: Set[str] = set()
        self.project_root = self._find_project_root()
        
        # Debounce mechanism to avoid excessive updates
        self.debounce_timers: Dict[str, threading.Timer] = {}
        self.debounce_delay = 0.5  # 500ms debounce
        
    def watch_directory(self, directory: str):
        """Watch a directory for file changes"""
        if os.path.isdir(directory):
            self.observer.schedule(self, directory, recursive=True)
            # Add all Python files in the directory to watched_files
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py'):
                        self.watched_files.add(os.path.join(root, file))
    
    def _find_project_root(self) -> str:
        """Find the project root directory"""
        current_dir = os.getcwd()
        # Look for common project markers
        while current_dir != os.path.dirname(current_dir):
            if any(os.path.exists(os.path.join(current_dir, marker)) 
                   for marker in ['.git', 'pyproject.toml', 'setup.py', 'requirements.txt']):
                return current_dir
            current_dir = os.path.dirname(current_dir)
        return os.getcwd()  # Fallback to current directory
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
            
        if not event.src_path.endswith('.py'):
            return
            
        # Debounce the file change event
        if event.src_path in self.debounce_timers:
            self.debounce_timers[event.src_path].cancel()
        
        timer = threading.Timer(self.debounce_delay, self._handle_file_change, args=[event.src_path])
        self.debounce_timers[event.src_path] = timer
        timer.start()
    
    def _handle_file_change(self, file_path: str):
        """Handle the actual file change after debounce"""
        # Remove the timer from the dictionary
        if file_path in self.debounce_timers:
            del self.debounce_timers[file_path]
        
        # Invalidate cache entries related to this file
        self._invalidate_cache_for_file(file_path)
        
        # Notify callback if provided
        if self.on_change_callback:
            self.on_change_callback(file_path)
    
    def _invalidate_cache_for_file(self, file_path: str):
        """Invalidate cache entries for a specific file"""
        # For now, we'll clear all cache entries that involve this file
        # In a more sophisticated implementation, we could be more selective
        keys_to_remove = []
        
        for key in self.cache.cache:
            # Check if the cache key involves the changed file
            # This is a simplified approach - in practice, you'd have a better way to track dependencies
            if file_path in key:
                keys_to_remove.append(key)
        
        # Remove the invalidated entries
        for key in keys_to_remove:
            if key in self.cache.cache:
                del self.cache.cache[key]
                if key in self.cache.access_order:
                    self.cache.access_order.remove(key)
    
    def start_watching(self):
        """Start watching for file changes"""
        self.observer.start()
    
    def stop_watching(self):
        """Stop watching for file changes"""
        self.observer.stop()
        self.observer.join()
        
        # Cancel any pending timers
        for timer in self.debounce_timers.values():
            timer.cancel()
        self.debounce_timers.clear()


def setup_context_watcher(index_dir: str, cache: ContextCache) -> ContextFileWatcher:
    """Set up a context watcher for real-time updates"""
    watcher = ContextFileWatcher(index_dir, cache)
    
    # Watch the project directory
    project_dir = os.path.dirname(index_dir)  # The index dir is typically .semindex in project root
    watcher.watch_directory(project_dir)
    
    return watcher