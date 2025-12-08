"""
Context Enhancement Module for semindex
Provides enhanced context extraction with documentation, comments, 
type information, and structural analysis for code generation.
"""
import ast
import re
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path


class DocumentationExtractor:
    """Extract documentation and comments from code for context"""
    
    @staticmethod
    def extract_docstrings_and_comments(source: str) -> Dict[str, Any]:
        """Extract docstrings and comments from source code"""
        tree = ast.parse(source)
        docstrings = []
        comments = []
        
        # Extract top-level docstrings
        if ast.get_docstring(tree):
            docstrings.append({
                'content': ast.get_docstring(tree),
                'line_start': 1,
                'line_end': 1,
                'type': 'module'
            })
        
        # Walk through all nodes to extract docstrings and comments
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                if docstring:
                    docstrings.append({
                        'content': docstring,
                        'line_start': node.lineno,
                        'line_end': node.lineno + len(docstring.split('\n')) - 1,
                        'type': type(node).__name__.lower(),
                        'name': getattr(node, 'name', 'module')
                    })
        
        # Extract comments using regex
        lines = source.split('\n')
        for i, line in enumerate(lines):
            comment_match = re.search(r'#\s*(.+)$', line)
            if comment_match:
                comments.append({
                    'content': comment_match.group(1).strip(),
                    'line': i + 1,
                    'full_line': line.strip()
                })
        
        return {
            'docstrings': docstrings,
            'comments': comments
        }
    
    @staticmethod
    def get_docstring_around_line(source: str, line_no: int, context_lines: int = 2) -> Optional[str]:
        """Get docstrings near a specific line"""
        tree = ast.parse(source)
        # Find the node containing the specified line
        for node in ast.walk(tree):
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                if node.lineno <= line_no <= node.end_lineno:
                    docstring = ast.get_docstring(node)
                    if docstring:
                        return docstring
        return None


class TypeExtractor:
    """Extract type information from code for context"""
    
    @staticmethod
    def extract_type_annotations(source: str) -> Dict[str, Any]:
        """Extract type annotations from source code"""
        tree = ast.parse(source)
        type_annotations = {
            'function_args': [],
            'function_returns': [],
            'variables': [],
            'imports': []
        }
        
        # Walk through the AST to find type annotations
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract argument annotations
                for arg in node.args.args:
                    if arg.annotation:
                        type_annotations['function_args'].append({
                            'function': node.name,
                            'arg_name': arg.arg,
                            'annotation': ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation),
                            'line': arg.lineno
                        })
                
                # Extract return annotation
                if node.returns:
                    type_annotations['function_returns'].append({
                        'function': node.name,
                        'return_type': ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns),
                        'line': node.returns.lineno if hasattr(node.returns, 'lineno') else node.lineno
                    })
            
            elif isinstance(node, ast.AnnAssign):
                # Extract variable annotations
                target_name = ''
                if isinstance(node.target, ast.Name):
                    target_name = node.target.id
                elif hasattr(node.target, 'attr'):
                    target_name = node.target.attr
                
                if target_name:
                    type_annotations['variables'].append({
                        'variable': target_name,
                        'annotation': ast.unparse(node.annotation) if hasattr(ast, 'unparse') else str(node.annotation),
                        'line': node.lineno
                    })
        
        # Extract type-related imports (typing module, etc.)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module in ('typing', 'typing_extensions'):
                    for alias in node.names:
                        type_annotations['imports'].append({
                            'import': alias.name,
                            'module': node.module,
                            'line': node.lineno
                        })
        
        return type_annotations


class StructureAnalyzer:
    """Analyze code structure for fine-grained understanding"""
    
    @staticmethod
    def analyze_code_structure(source: str) -> Dict[str, Any]:
        """Analyze the structural elements of code"""
        tree = ast.parse(source)
        
        structure = {
            'classes': [],
            'functions': [],
            'imports': [],
            'assignments': [],
            'control_flow': [],
            'scopes': []
        }
        
        # Analyze classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append({
                            'name': item.name,
                            'lineno': item.lineno,
                            'is_async': isinstance(item, ast.AsyncFunctionDef)
                        })
                
                structure['classes'].append({
                    'name': node.name,
                    'lineno': node.lineno,
                    'end_lineno': getattr(node, 'end_lineno', node.lineno),
                    'methods': methods,
                    'bases': [ast.unparse(base) if hasattr(ast, 'unparse') else str(base) for base in node.bases]
                })
        
        # Analyze functions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [arg.arg for arg in node.args.args if arg.arg != 'self']
                
                structure['functions'].append({
                    'name': node.name,
                    'lineno': node.lineno,
                    'end_lineno': getattr(node, 'end_lineno', node.lineno),
                    'args': args,
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'is_method': StructureAnalyzer._is_method(node)
                })
        
        # Analyze imports
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        structure['imports'].append({
                            'name': alias.name,
                            'alias': alias.asname,
                            'type': 'import',
                            'lineno': node.lineno
                        })
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        structure['imports'].append({
                            'name': f"{node.module}.{alias.name}",
                            'alias': alias.asname,
                            'type': 'from_import',
                            'lineno': node.lineno
                        })
        
        return structure
    
    @staticmethod
    def _is_method(node: ast.AST) -> bool:
        """Check if a function node is a method (has 'self' as first argument)"""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.args.args:
                return node.args.args[0].arg == 'self'
        return False
    
    @staticmethod
    def get_scope_at_line(source: str, line_no: int) -> Dict[str, Any]:
        """Determine the code scope (function, class, module) at a specific line"""
        tree = ast.parse(source)
        
        # Find the innermost scope containing the line
        scopes = []
        
        # Walk through all nodes and find those that contain the line
        for node in ast.walk(tree):
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                if node.lineno <= line_no <= getattr(node, 'end_lineno', node.lineno):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        scopes.append({
                            'type': 'function',
                            'name': node.name,
                            'start_line': node.lineno,
                            'end_line': getattr(node, 'end_lineno', node.lineno),
                            'is_async': isinstance(node, ast.AsyncFunctionDef)
                        })
                    elif isinstance(node, ast.ClassDef):
                        scopes.append({
                            'type': 'class',
                            'name': node.name,
                            'start_line': node.lineno,
                            'end_line': getattr(node, 'end_lineno', node.lineno)
                        })
        
        # Return the innermost scope (with smallest range that contains the line)
        if scopes:
            # Sort by range size (end - start), smallest first
            scopes.sort(key=lambda x: x['end_line'] - x['start_line'])
            return scopes[0]
        
        # If no specific scope, return module scope
        return {
            'type': 'module',
            'name': 'module',
            'start_line': 1,
            'end_line': source.count('\n') + 1
        }


def enhance_context_with_multi_modal_data(file_path: str, line_no: int) -> Dict[str, Any]:
    """Enhance context with multi-modal information"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Extract documentation and comments
        doc_extractor = DocumentationExtractor()
        doc_data = doc_extractor.extract_docstrings_and_comments(source)
        docstring_at_line = doc_extractor.get_docstring_around_line(source, line_no)
        
        # Extract type information
        type_extractor = TypeExtractor()
        type_data = type_extractor.extract_type_annotations(source)
        
        # Analyze code structure
        struct_analyzer = StructureAnalyzer()
        struct_data = struct_analyzer.analyze_code_structure(source)
        scope_at_line = struct_analyzer.get_scope_at_line(source, line_no)
        
        return {
            'documentation': doc_data,
            'docstring_at_line': docstring_at_line,
            'types': type_data,
            'structure': struct_data,
            'scope': scope_at_line
        }
    except Exception as e:
        return {'error': str(e)}