"""
Pattern Analysis Module for semindex
Provides pattern extraction, matching, and template generation for code generation.
"""
import os
import ast
import re
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from .search import Searcher
from .store import DB_NAME, db_conn
from .embed import Embedder


class PatternExtractor:
    """Extract common patterns from the codebase"""
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.searcher = Searcher(index_dir)
        self.embedder = Embedder()
    
    def extract_common_patterns(self, top_k: int = 20) -> List[Dict[str, Any]]:
        """Extract common patterns from the indexed codebase"""
        patterns = []
        
        # Get frequently occurring code snippets from the index
        # This is a simplified approach - in practice, you'd want more sophisticated pattern detection
        try:
            # Query for common patterns based on semantic similarity
            common_queries = [
                "function definition",
                "class definition", 
                "error handling",
                "file operations",
                "database queries",
                "API endpoints",
                "logging patterns",
                "configuration patterns"
            ]
            
            for query in common_queries:
                results = self.searcher.query(query, top_k=3)
                for score, symbol_id, symbol_info in results:
                    path, name, kind, start_line, end_line, signature = symbol_info[:6]
                    
                    # Get the actual code content
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            lines = content.splitlines()
                            code_snippet = "\n".join(lines[start_line-1:end_line])
                            
                            patterns.append({
                                'name': name,
                                'type': kind,
                                'path': path,
                                'code': code_snippet,
                                'signature': signature,
                                'score': score,
                                'pattern_type': self._categorize_pattern(code_snippet, signature)
                            })
                    except Exception:
                        continue  # Skip if file reading fails
            
            # Remove duplicates and return top patterns
            unique_patterns = []
            seen_patterns = set()
            
            for pattern in patterns:
                pattern_key = (pattern['name'], pattern['code'][:100])  # Use first 100 chars as key
                if pattern_key not in seen_patterns:
                    seen_patterns.add(pattern_key)
                    unique_patterns.append(pattern)
            
            return unique_patterns[:top_k]
            
        except Exception as e:
            print(f"Error extracting patterns: {e}")
            return []
    
    def _categorize_pattern(self, code: str, signature: str) -> str:
        """Categorize a code pattern based on its content"""
        code_lower = code.lower()
        signature_lower = signature.lower()
        
        if 'def test_' in code_lower or 'assert' in code_lower:
            return 'test'
        elif 'exception' in code_lower or 'error' in code_lower or 'try:' in code_lower or 'except' in code_lower:
            return 'error_handling'
        elif 'open(' in code_lower or '.read(' in code_lower or '.write(' in code_lower:
            return 'file_operation'
        elif 'class' in code_lower and ('api' in code_lower or 'handler' in code_lower or 'controller' in code_lower):
            return 'api_endpoint'
        elif 'logger' in code_lower or 'logging' in code_lower:
            return 'logging'
        elif 'config' in code_lower or 'setting' in code_lower:
            return 'configuration'
        elif 'for' in code_lower or 'while' in code_lower:
            return 'loop'
        elif 'if' in code_lower or 'elif' in code_lower:
            return 'conditional'
        else:
            return 'general'


class PatternMatcher:
    """Match current context to known patterns"""
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.extractor = PatternExtractor(index_dir)
        self.known_patterns = self.extractor.extract_common_patterns()
    
    def find_matching_patterns(self, target_context: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find patterns that match the current target context"""
        if not self.known_patterns:
            return []
        
        # Simple approach: match based on keyword similarity
        # In a more advanced system, you'd use embeddings or AST similarity
        target_lower = target_context.lower()
        matches = []
        
        for pattern in self.known_patterns:
            # Calculate a simple relevance score
            score = 0
            pattern_lower = pattern['code'].lower()
            
            # Match based on keywords in the signature
            if pattern['signature']:
                sig_words = pattern['signature'].lower().split()
                for word in sig_words:
                    if len(word) > 2:  # Skip short words
                        if word in target_lower:
                            score += 1
            
            # Match based on pattern type
            if pattern['type'].lower() in target_lower:
                score += 2
                
            # Match based on content similarity
            common_tokens = set(target_lower.split()) & set(pattern_lower.split())
            score += len(common_tokens) * 0.1
            
            if score > 0:
                matches.append({
                    'pattern': pattern,
                    'relevance_score': score
                })
        
        # Sort by relevance and return top matches
        matches.sort(key=lambda x: x['relevance_score'], reverse=True)
        return matches[:top_k]


class TemplateGenerator:
    """Generate templates from discovered patterns"""
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.extractor = PatternExtractor(index_dir)
    
    def generate_templates(self) -> Dict[str, List[str]]:
        """Generate templates based on extracted patterns"""
        patterns = self.extractor.extract_common_patterns(top_k=50)
        
        templates = {}
        
        # Group patterns by type
        for pattern in patterns:
            pattern_type = pattern.get('pattern_type', 'general')
            
            if pattern_type not in templates:
                templates[pattern_type] = []
            
            # Create a template from the pattern
            template = self._create_template_from_pattern(pattern['code'])
            if template and template not in templates[pattern_type]:
                templates[pattern_type].append(template)
        
        return templates
    
    def _create_template_from_pattern(self, code: str) -> Optional[str]:
        """Create a template from a pattern by replacing specific values with placeholders"""
        # This is a basic implementation - in practice, you'd want more sophisticated template extraction
        try:
            # Parse the code to identify variables, strings, and other replaceable elements
            tree = ast.parse(code)
            
            # For now, return the original code as a template
            # In a more advanced implementation, you'd replace specific elements with placeholders
            return code
        except SyntaxError:
            # If parsing fails, return None
            return None


class TemplateRegistry:
    """Registry for storing and managing code templates"""
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.templates = {}
        self.pattern_matcher = PatternMatcher(index_dir)
        self.template_generator = TemplateGenerator(index_dir)
        
        # Load initial templates
        self.refresh_templates()
    
    def refresh_templates(self):
        """Refresh the template registry with current patterns"""
        self.templates = self.template_generator.generate_templates()
    
    def get_relevant_templates(self, context: str, top_k: int = 3) -> List[str]:
        """Get templates relevant to the current context"""
        # Find matching patterns first
        matches = self.pattern_matcher.find_matching_patterns(context, top_k*2)
        
        relevant_templates = []
        for match in matches:
            pattern_type = match['pattern']['pattern_type']
            if pattern_type in self.templates:
                # Add templates of matching types
                for template in self.templates[pattern_type]:
                    if template not in relevant_templates:
                        relevant_templates.append(template)
        
        return relevant_templates[:top_k]
    
    def add_template(self, template: str, category: str = 'general'):
        """Add a new template to the registry"""
        if category not in self.templates:
            self.templates[category] = []
        
        if template not in self.templates[category]:
            self.templates[category].append(template)


def analyze_patterns_and_templates(user_request: str, index_dir: str) -> Dict[str, Any]:
    """Analyze patterns and templates relevant to a user request"""
    try:
        matcher = PatternMatcher(index_dir)
        matches = matcher.find_matching_patterns(user_request)
        
        registry = TemplateRegistry(index_dir)
        relevant_templates = registry.get_relevant_templates(user_request)
        
        return {
            'matching_patterns': matches,
            'relevant_templates': relevant_templates,
            'pattern_count': len(matches),
            'template_count': len(relevant_templates)
        }
    except Exception as e:
        return {
            'error': str(e),
            'matching_patterns': [],
            'relevant_templates': [],
            'pattern_count': 0,
            'template_count': 0
        }