"""
Context Generation Module for semindex
Provides context-aware code generation capabilities by aggregating information
from multiple levels in the codebase.
"""
import os
import ast
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from .search import Searcher
from .ast_py import parse_python_symbols
from .model import Symbol
from .embed import Embedder
from .local_llm import LocalLLM
from .store import DB_NAME, db_conn
from .context_enhancer import DocumentationExtractor, TypeExtractor, StructureAnalyzer, enhance_context_with_multi_modal_data
from .intent_analyzer import analyze_intent_and_decompose, IntentType
from .pattern_analyzer import analyze_patterns_and_templates, TemplateRegistry


class Context:
    """Represents aggregated context for code generation"""
    def __init__(self, 
                 local: Optional[str] = None,
                 enhanced: Optional[Dict] = None,
                 file: Optional[Dict] = None,
                 cross_file: Optional[List[Dict]] = None,
                 project: Optional[Dict] = None):
        self.local = local
        self.enhanced = enhanced or {}  # New enhanced context
        self.file = file or {}
        self.cross_file = cross_file or []
        self.project = project or {}


class GeneratedCode:
    """Represents generated code with metadata"""
    def __init__(self, code: str, context_used: Context, validation_result: Dict[str, Any]):
        self.code = code
        self.context_used = context_used
        self.validation_result = validation_result


class ContextAggregator:
    """Aggregates context from multiple levels for code generation"""
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.searcher = Searcher(index_dir)
        self.embedder = Embedder()
        
    def get_context_for_position(self, 
                                 file_path: str, 
                                 line_no: int, 
                                 character_pos: int = 0) -> Context:
        """
        Aggregate context from multiple levels around a specific position in a file
        """
        # 1. Local context: Current function/class definition
        local_context = self.get_local_context(file_path, line_no)
        
        # 1.5. Enhanced multi-modal context: Documentation, types, structure
        enhanced_context = self.get_enhanced_context(file_path, line_no)
        
        # 2. File context: Imports, global definitions
        file_context = self.get_file_context(file_path)
        
        # 3. Cross-file context: Related symbols and dependencies
        cross_file_context = self.get_related_context(file_path, local_context)
        
        # 4. Project context: Architecture patterns, common utilities
        project_context = self.get_project_context(file_path)
        
        return Context(
            local=local_context,
            enhanced=enhanced_context,  # New enhanced context
            file=file_context,
            cross_file=cross_file_context,
            project=project_context
        )
    
    def get_enhanced_context(self, file_path: str, line_no: int) -> Dict[str, Any]:
        """Get enhanced multi-modal context including documentation, types, and structure"""
        return enhance_context_with_multi_modal_data(file_path, line_no)
    
    def get_local_context(self, file_path: str, line_no: int) -> str:
        """Extract AST node-based context around the specific line in the file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
                lines = f.readlines()
            
            # First, try to get context based on AST node scope
            tree = ast.parse(source)
            
            # Find the specific AST node containing the line
            target_node = self._find_node_at_line(tree, line_no)
            
            if target_node:
                # Get the exact scope containing this line
                if isinstance(target_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Get the full function context
                    start_pos = target_node.lineno - 1  # Convert to 0-indexed
                    end_pos = min(getattr(target_node, 'end_lineno', target_node.lineno), len(lines))
                    node_context = lines[start_pos:end_pos]
                    return "".join(node_context)
                elif isinstance(target_node, ast.ClassDef):
                    # Get the full class context
                    start_pos = target_node.lineno - 1
                    end_pos = min(getattr(target_node, 'end_lineno', target_node.lineno), len(lines))
                    node_context = lines[start_pos:end_pos]
                    return "".join(node_context)
                elif isinstance(target_node, ast.stmt):  # Statement
                    # Get surrounding context around the statement
                    start_pos = max(0, target_node.lineno - 6)  # 5 lines before
                    end_pos = min(len(lines), target_node.lineno + 5)  # 5 lines after
                    node_context = lines[start_pos:end_pos]
                    return "".join(node_context)
            
            # Fallback to original approach if AST node not found
            start_line = max(0, line_no - 11)  # 0-indexed, subtract 1 for 0-based
            end_line = min(len(lines), line_no + 10)  # 0-indexed
            
            context_lines = lines[start_line:end_line]
            return "".join(context_lines)
        except Exception:
            return ""
    
    def _find_node_at_line(self, tree: ast.AST, line_no: int) -> Optional[ast.AST]:
        """Find the AST node that contains the specific line number"""
        # Find nodes that contain the target line number
        for node in ast.walk(tree):
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                if node.lineno <= line_no <= getattr(node, 'end_lineno', node.lineno):
                    # Return the smallest node that contains the line
                    # (this will be the most specific context)
                    return node
        return None
    
    def get_file_context(self, file_path: str) -> Dict[str, Any]:
        """Extract file-level context like imports, classes, functions"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
                
            # Parse AST to get file structure
            tree = ast.parse(source)
            
            imports = []
            classes = []
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        imports.extend([alias.name for alias in node.names])
                    elif isinstance(node, ast.ImportFrom):
                        imports.append(f"from {node.module} import ...")
                
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        'name': node.name,
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        'lineno': node.lineno
                    })
                
                elif isinstance(node, ast.FunctionDef):
                    if node.name not in [f['name'] for f in functions]:  # Avoid duplicates
                        functions.append({
                            'name': node.name,
                            'lineno': node.lineno,
                            'args': [arg.arg for arg in node.args.args if arg.arg != 'self']
                        })
            
            return {
                'imports': imports,
                'classes': classes,
                'functions': functions,
                'file_path': file_path
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_related_context(self, file_path: str, local_context: str) -> List[Dict]:
        """Get context from related files based on semantic similarity and call relationships"""
        try:
            # Use the searcher to find semantically related code
            results = self.searcher.query(
                query=f"code related to {' '.join(local_context.split()[:50])}",  # Limit query size
                top_k=5
            )
            
            related_contexts = []
            for score, _symbol_id, symbol_info in results:
                path, name, kind, start_line, end_line, signature = symbol_info[:6]
                
                # Skip if it's the same file to avoid duplication
                if path == file_path:
                    continue
                    
                related_contexts.append({
                    'path': path,
                    'name': name,
                    'kind': kind,
                    'signature': signature,
                    'score': score,
                    'relationship': 'semantic'  # Default relationship type
                })
            
            # Enhance with call graph relationships
            enhanced_contexts = self._add_call_graph_context(file_path, related_contexts)
            
            return enhanced_contexts
        except Exception:
            return []
    
    def _add_call_graph_context(self, file_path: str, related_contexts: List[Dict]) -> List[Dict]:
        """Add call graph context to related contexts"""
        try:
            db_path = os.path.join(self.index_dir, DB_NAME)
            with db_conn(db_path) as con:
                cursor = con.cursor()
                
                # Get all symbol IDs in the target file
                cursor.execute(
                    "SELECT id, name FROM symbols WHERE path = ?",
                    (file_path,)
                )
                file_symbols = cursor.fetchall()
                
                # For each symbol in the target file, find callers and callees
                for symbol_id, symbol_name in file_symbols:
                    # Get callers (who calls this symbol)
                    cursor.execute(
                        "SELECT c.caller_id, s.name, s.path FROM calls c "
                        "JOIN symbols s ON c.caller_id = s.id "
                        "WHERE c.callee_id = ?",
                        (symbol_id,)
                    )
                    callers = cursor.fetchall()
                    
                    for caller_id, caller_name, caller_path in callers:
                        if caller_path != file_path:  # Don't duplicate same-file relations
                            related_contexts.append({
                                'path': caller_path,
                                'name': caller_name,
                                'kind': 'caller',
                                'signature': f'calls {symbol_name}',
                                'score': 0.9,  # High relevance for direct callers
                                'relationship': 'calls_this_symbol'
                            })
                    
                    # Get callees (what this symbol calls)
                    cursor.execute(
                        "SELECT c.callee_id, s.name, s.path FROM calls c "
                        "JOIN symbols s ON c.callee_id = s.id "
                        "WHERE c.caller_id = ?",
                        (symbol_id,)
                    )
                    callees = cursor.fetchall()
                    
                    for callee_id, callee_name, callee_path in callees:
                        if callee_path != file_path:  # Don't duplicate same-file relations
                            related_contexts.append({
                                'path': callee_path,
                                'name': callee_name,
                                'kind': 'callee',
                                'signature': f'called_by {symbol_name}',
                                'score': 0.8,  # High relevance for called symbols
                                'relationship': 'called_by_this_symbol'
                            })
                        
            # Remove duplicates while preserving order
            seen = set()
            unique_contexts = []
            for ctx in related_contexts:
                ctx_key = (ctx['path'], ctx['name'])
                if ctx_key not in seen:
                    seen.add(ctx_key)
                    unique_contexts.append(ctx)
            
            return unique_contexts
        except Exception:
            # If call graph query fails, return original contexts
            return related_contexts
    
    def get_project_context(self, file_path: str) -> Dict[str, Any]:
        """Get project-level context like architecture patterns, common utilities"""
        try:
            # Find common patterns in the project
            project_dir = os.path.dirname(file_path)
            
            # Look for common architectural elements
            common_utils = self.searcher.query("common utilities", top_k=3)
            patterns = self.searcher.query("design patterns", top_k=3)
            
            project_info = {
                'common_utilities': [],
                'design_patterns': [],
                'project_root': project_dir
            }
            
            for score, _symbol_id, symbol_info in common_utils:
                path, name, kind, start_line, end_line, signature = symbol_info[:6]
                project_info['common_utilities'].append({
                    'path': path,
                    'name': name,
                    'kind': kind,
                    'signature': signature
                })
                
            for score, _symbol_id, symbol_info in patterns:
                path, name, kind, start_line, end_line, signature = symbol_info[:6]
                project_info['design_patterns'].append({
                    'path': path,
                    'name': name,
                    'kind': kind,
                    'signature': signature
                })
            
            return project_info
        except Exception:
            return {}


class ContextCache:
    """Caches context to improve performance"""
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []  # For LRU eviction
        self.file_dependencies = {}  # Track which cache entries depend on which files
    
    def get_cached_context(self, cache_key: str) -> Optional[Context]:
        """Retrieve cached context if available"""
        if cache_key in self.cache:
            # Move to end (most recently used)
            if cache_key in self.access_order:
                self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            return self.cache[cache_key]
        return None
    
    def cache_context(self, cache_key: str, context: Context, file_dependencies: Optional[List[str]] = None):
        """Cache the context with optional file dependencies for invalidation"""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            if self.access_order:
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
                if lru_key in self.file_dependencies:
                    del self.file_dependencies[lru_key]
        
        self.cache[cache_key] = context
        self.access_order.append(cache_key)
        
        if file_dependencies:
            self.file_dependencies[cache_key] = set(file_dependencies)
    
    def invalidate_for_file(self, file_path: str):
        """Invalidate cache entries that depend on a specific file"""
        keys_to_remove = []
        for cache_key, dependencies in self.file_dependencies.items():
            if file_path in dependencies and cache_key in self.cache:
                keys_to_remove.append(cache_key)
        
        for key in keys_to_remove:
            del self.cache[key]
            self.access_order = [k for k in self.access_order if k != key]
            if key in self.file_dependencies:
                del self.file_dependencies[key]
    
    def clear_all(self):
        """Clear all cached contexts"""
        self.cache.clear()
        self.access_order.clear()
        self.file_dependencies.clear()


def generate_code_with_context(
    index_dir: str,
    file_path: str,
    line_no: int,
    user_request: str,
    max_context_length: int = 2048
) -> str:
    """
    Generate code with context from multiple levels of the codebase
    """
    # Analyze user intent to adapt context and generation
    intent_analysis = analyze_intent_and_decompose(user_request, index_dir)
    intent = intent_analysis['intent']
    
    # Initialize context aggregator with caching
    aggregator = ContextAggregator(index_dir)
    cache = ContextCache()
    
    # Create cache key based on the request and position
    cache_key = hashlib.md5(f"{file_path}_{line_no}_{user_request}".encode()).hexdigest()
    
    # Check cache first
    context = cache.get_cached_context(cache_key)
    if not context:
        context = aggregator.get_context_for_position(file_path, line_no, 0)
        # Track file dependencies for cache invalidation
        file_deps = [file_path]  # Add main file
        # Add related files from cross-file context
        for item in context.cross_file:
            if 'path' in item:
                file_deps.append(item['path'])
        cache.cache_context(cache_key, context, file_dependencies=file_deps)
    
    # Format context for the LLM, adapted based on intent
    formatted_context = format_context_for_llm(context)
    
    # Build generation prompt adapted to intent
    system_prompt = _build_system_prompt_for_intent(intent)
    
    user_prompt = _build_user_prompt_for_intent(
        user_request, 
        formatted_context, 
        file_path, 
        line_no, 
        intent
    )
    
    # Generate code using LocalLLM
    llm = LocalLLM(
        model_type="transformer",
        model_name=os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    )
    
    generated_code = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=512,
        context_chunks=[formatted_context] if formatted_context else None
    )
    
    return generated_code


def _build_system_prompt_for_intent(intent: IntentType) -> str:
    """Build system prompt based on the detected intent"""
    base_prompt = "You are an expert code assistant. Generate code based on the user's request while maintaining consistency with the existing codebase. Follow the patterns and conventions shown in the context."
    
    if intent == IntentType.IMPLEMENTATION:
        return base_prompt + " Focus on implementing the requested functionality following best practices."
    elif intent == IntentType.REFACTORING:
        return base_prompt + " Focus on improving code structure, readability, and performance while maintaining the same functionality."
    elif intent == IntentType.DEBUGGING:
        return base_prompt + " Focus on identifying and fixing issues in the code. Consider common error patterns and best practices."
    elif intent == IntentType.DOCUMENTATION:
        return base_prompt + " Focus on generating clear, comprehensive documentation that explains the purpose, parameters, return values, and usage."
    elif intent == IntentType.TESTING:
        return base_prompt + " Focus on generating comprehensive tests that cover normal cases, edge cases, and error conditions."
    else:
        return base_prompt


def _build_user_prompt_for_intent(
    user_request: str, 
    formatted_context: str, 
    file_path: str, 
    line_no: int, 
    intent: IntentType
) -> str:
    """Build user prompt based on the detected intent"""
    if intent == IntentType.DOCUMENTATION:
        return (
            f"Request: {user_request}\n\n"
            f"Context:\n{formatted_context}\n\n"
            f"Generate appropriate documentation for line {line_no} in {file_path}:"
        )
    elif intent == IntentType.TESTING:
        return (
            f"Request: {user_request}\n\n"
            f"Code context:\n{formatted_context}\n\n"
            f"Generate appropriate tests for the functionality in {file_path}:"
        )
    elif intent == IntentType.DEBUGGING:
        return (
            f"Request: {user_request}\n\n"
            f"Code context:\n{formatted_context}\n\n"
            f"Analyze and fix issues for line {line_no} in {file_path}:"
        )
    else:
        return (
            f"Request: {user_request}\n\n"
            f"Context:\n{formatted_context}\n\n"
            f"Generate appropriate code for line {line_no} in {file_path}:"
        )


class InteractiveCodeRefiner:
    """Enables iterative refinement of generated code through user feedback"""
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.generator = ContextAwareCodeGenerator(index_dir)
        self.execution_generator = ExecutionGuidedGenerator(index_dir)  # New execution-guided generator
        self.conversation_memory = []  # Track conversation history
    
    def refine_with_feedback(self, original_code: str, feedback: str, 
                           file_path: str, line_no: int, request: str) -> GeneratedCode:
        """Refine generated code based on user feedback"""
        # Add the interaction to conversation memory
        self.conversation_memory.append({
            'original_request': request,
            'generated_code': original_code,
            'user_feedback': feedback
        })
        
        # Build a refined request incorporating the feedback
        refined_request = self._build_refined_request(request, feedback, original_code)
        
        # Generate refined code
        result = self.execution_generator.generate_with_execution_feedback(
            file_path, line_no, refined_request
        )
        
        return result
    
    def _build_refined_request(self, original_request: str, feedback: str, original_code: str) -> str:
        """Build a refined request that incorporates user feedback"""
        return (
            f"Original request: {original_request}\n\n"
            f"Generated code: {original_code}\n\n"
            f"User feedback: {feedback}\n\n"
            f"Please generate improved code based on this feedback."
        )
    
    def start_conversation(self) -> None:
        """Initialize a new conversation for code refinement"""
        self.conversation_memory = []
    
    def get_conversation_context(self) -> str:
        """Get the conversation context for LLM prompting"""
        if not self.conversation_memory:
            return ""
        
        context_parts = ["Previous conversation:"]
        for i, interaction in enumerate(self.conversation_memory[-3:], 1):  # Include last 3 interactions
            context_parts.append(f"Interaction {i}:")
            context_parts.append(f"  Request: {interaction['original_request']}")
            context_parts.append(f"  Code: {interaction['generated_code'][:200]}...")  # Truncate code
            context_parts.append(f"  Feedback: {interaction['user_feedback']}")
            context_parts.append("")
        
        return "\n".join(context_parts)


def format_context_for_llm(context: Context) -> str:
    """Format context in a way that's useful for LLM generation"""
    parts = []
    
    if context.local:
        parts.append(f"LOCAL CONTEXT (around target position):\n{context.local[:500]}")
    
    # Enhanced multi-modal context
    if context.enhanced:
        if context.enhanced.get('documentation'):
            docs = context.enhanced['documentation']
            if docs.get('docstrings'):
                docstrings = [d['content'] for d in docs['docstrings'] if d['content']]
                if docstrings:
                    parts.append(f"RELEVANT DOCSTRINGS: {'; '.join(docstrings[:3])}")
            if docs.get('comments'):
                comments = [c['content'] for c in docs['comments']]
                if comments:
                    parts.append(f"RELEVANT COMMENTS: {'; '.join(comments[:5])}")
        
        if context.enhanced.get('types'):
            types = context.enhanced['types']
            if types.get('function_args'):
                args = [f"{t['function']}({t['arg_name']}: {t['annotation']})" for t in types['function_args']]
                if args:
                    parts.append(f"TYPE ANNOTATIONS (function args): {', '.join(args[:5])}")
            if types.get('function_returns'):
                returns = [f"{t['function']} -> {t['return_type']}" for t in types['function_returns']]
                if returns:
                    parts.append(f"TYPE ANNOTATIONS (function returns): {', '.join(returns[:5])}")
        
        if context.enhanced.get('structure'):
            struct = context.enhanced['structure']
            if struct.get('classes'):
                class_names = [c['name'] for c in struct['classes']]
                parts.append(f"CLASSES IN FILE: {', '.join(class_names)}")
            if struct.get('functions'):
                func_names = [f['name'] for f in struct['functions']]
                parts.append(f"FUNCTIONS IN FILE: {', '.join(func_names)}")
        
        if context.enhanced.get('scope'):
            scope = context.enhanced['scope']
            parts.append(f"CURRENT SCOPE: {scope['type']} {scope['name']} (lines {scope['start_line']}-{scope['end_line']})")
    
    if context.file:
        if 'imports' in context.file:
            parts.append(f"FILE IMPORTS: {', '.join(context.file['imports'][:10])}")
        if 'classes' in context.file:
            class_names = [c['name'] for c in context.file['classes']]
            parts.append(f"FILE CLASSES: {', '.join(class_names)}")
        if 'functions' in context.file:
            func_names = [f['name'] for f in context.file['functions']]
            parts.append(f"FILE FUNCTIONS: {', '.join(func_names)}")
    
    if context.cross_file:
        related = [f"{item['name']} ({item['kind']}) in {item['path']}" for item in context.cross_file[:3]]
        parts.append(f"RELATED CONTEXT: {', '.join(related)}")
    
    if context.project:
        if context.project.get('common_utilities'):
            utils = [u['name'] for u in context.project['common_utilities'][:3]]
            parts.append(f"PROJECT UTILITIES: {', '.join(utils)}")
    
    return "\n\n".join(parts)


class CodeValidator:
    """Validates generated code for syntax, type, and quality issues"""
    
    def __init__(self, index_dir: str = None):
        self.index_dir = index_dir
    
    def validate_generated_code(self, code: str, language: str = "python", 
                               original_context: str = "", file_path: str = "") -> Dict[str, Any]:
        """Validate that generated code follows syntax, type, and quality rules"""
        errors = []
        warnings = []
        
        if language == "python":
            # Syntax validation
            try:
                ast.parse(code)
            except SyntaxError as e:
                errors.append({
                    "type": "syntax_error",
                    "message": str(e),
                    "line": e.lineno if e.lineno else 0
                })
            except Exception as e:
                errors.append({
                    "type": "parse_error",
                    "message": f"Could not parse code: {str(e)}",
                    "line": 0
                })
            
            # Additional validations could go here
            # For example, checking for common anti-patterns, complexity, etc.
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "language": language
        }


class ExecutionGuidedGenerator:
    """Generator that refines code based on execution feedback"""
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.validator = CodeValidator(index_dir)
        self.generator = ContextAwareCodeGenerator(index_dir)
    
    def generate_with_execution_feedback(self, file_path: str, line_no: int, 
                                       request: str, max_iterations: int = 3) -> GeneratedCode:
        """Generate code with iterative refinement based on validation feedback"""
        current_iteration = 0
        last_generated_code = None
        last_validation_result = None
        
        while current_iteration < max_iterations:
            # Generate code using the base generator
            result = self.generator.generate_at_position(file_path, line_no, request)
            
            # Validate the generated code
            validation_result = self.validator.validate_generated_code(
                result.code, 
                "python", 
                result.context_used.local if result.context_used else "",
                file_path
            )
            
            # Check if the code passes validation
            if validation_result["is_valid"]:
                return GeneratedCode(
                    code=result.code,
                    context_used=result.context_used,
                    validation_result=validation_result
                )
            else:
                # If validation fails, try to generate improved code
                last_generated_code = result.code
                last_validation_result = validation_result
                
                # Update the request to include feedback for next iteration
                feedback = self._format_validation_feedback(validation_result)
                refined_request = f"{request}\n\nFeedback: {feedback}"
                
                current_iteration += 1
        
        # If we've exhausted iterations, return the last attempt
        return GeneratedCode(
            code=last_generated_code or "",
            context_used=result.context_used if 'result' in locals() else None,
            validation_result=last_validation_result or {"is_valid": False, "errors": [], "warnings": [], "language": "python"}
        )
    
    def _format_validation_feedback(self, validation_result: Dict) -> str:
        """Format validation feedback for use in refining the generation"""
        feedback_parts = []
        
        if validation_result["errors"]:
            errors = [f"{error['type']}: {error['message']}" for error in validation_result["errors"]]
            feedback_parts.append("Fix these errors: " + "; ".join(errors))
        
        if validation_result["warnings"]:
            warnings = [f"{warning['type']}: {warning['message']}" for warning in validation_result["warnings"]]
            feedback_parts.append("Consider these warnings: " + "; ".join(warnings))
        
        return " ".join(feedback_parts)


class FormattingEnforcer:
    """Applies project-specific formatting to generated code"""
    
    def format_code(self, code: str, file_path: str) -> str:
        """Apply formatting based on the target file's style"""
        # For now, just return the code as is
        # In the future, we could implement more sophisticated formatting
        # based on the style of the target file
        return code


# Define GeneratedCode early so other classes can reference it
class GeneratedCode:
    """Represents generated code with metadata"""
    def __init__(self, code: str, context_used: Context, validation_result: Dict[str, Any]):
        self.code = code
        self.context_used = context_used
        self.validation_result = validation_result


class InteractiveCodeRefiner:
    """Enables iterative refinement of generated code through user feedback"""
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.generator = ContextAwareCodeGenerator(index_dir)
        self.execution_generator = ExecutionGuidedGenerator(index_dir)  # New execution-guided generator
        self.conversation_memory = []  # Track conversation history
    
    def refine_with_feedback(self, original_code: str, feedback: str, 
                           file_path: str, line_no: int, request: str) -> GeneratedCode:
        """Refine generated code based on user feedback"""
        # Add the interaction to conversation memory
        self.conversation_memory.append({
            'original_request': request,
            'generated_code': original_code,
            'user_feedback': feedback
        })
        
        # Build a refined request incorporating the feedback
        refined_request = self._build_refined_request(request, feedback, original_code)
        
        # Generate refined code
        result = self.execution_generator.generate_with_execution_feedback(
            file_path, line_no, refined_request
        )
        
        return result
    
    def _build_refined_request(self, original_request: str, feedback: str, original_code: str) -> str:
        """Build a refined request that incorporates user feedback"""
        return (
            f"Original request: {original_request}\n\n"
            f"Generated code: {original_code}\n\n"
            f"User feedback: {feedback}\n\n"
            f"Please generate improved code based on this feedback."
        )
    
    def start_conversation(self) -> None:
        """Initialize a new conversation for code refinement"""
        self.conversation_memory = []
    
    def get_conversation_context(self) -> str:
        """Get the conversation context for LLM prompting"""
        if not self.conversation_memory:
            return ""
        
        context_parts = ["Previous conversation:"]
        for i, interaction in enumerate(self.conversation_memory[-3:], 1):  # Include last 3 interactions
            context_parts.append(f"Interaction {i}:")
            context_parts.append(f"  Request: {interaction['original_request']}")
            context_parts.append(f"  Code: {interaction['generated_code'][:200]}...")  # Truncate code
            context_parts.append(f"  Feedback: {interaction['user_feedback']}")
            context_parts.append("")
        
        return "\n".join(context_parts)


class ContextAwareCodeGenerator:
    """Main class for contextual code generation in semindex"""
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.context_aggregator = ContextAggregator(index_dir)
        self.generator = LocalLLM(
            model_type="transformer",
            model_name=os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        )
        self.validator = CodeValidator(index_dir)
        self.formatter = FormattingEnforcer()
        self.cache = ContextCache()
        self.template_registry = TemplateRegistry(index_dir)
    
    def generate_at_position(self, file_path: str, line_no: int, request: str) -> GeneratedCode:
        """Generate code at a specific position with full context awareness"""
        
        # Analyze user intent to adapt generation
        intent_analysis = analyze_intent_and_decompose(request, self.index_dir)
        intent = intent_analysis['intent']
        
        # Create cache key based on the request and position
        cache_key = f"{file_path}_{line_no}_{request[:50]}"  # Limit request length in cache key
        
        # Check cache first
        context = self.cache.get_cached_context(cache_key)
        if not context:
            # 1. Build comprehensive context
            context = self.context_aggregator.get_context_for_position(
                file_path, line_no, 0
            )
            self.cache.cache_context(cache_key, context)
        
        # Analyze patterns relevant to the request
        pattern_analysis = analyze_patterns_and_templates(request, self.index_dir)
        
        # 2. Generate code with context, intent, and patterns
        generated_code = self._generate_with_context_intent_and_patterns(
            context, request, intent, pattern_analysis
        )
        
        # 3. Validate and format
        validation_result = self.validator.validate_generated_code(generated_code)
        if not validation_result["is_valid"]:
            # Try to provide a basic fix suggestion (in a real implementation, this would be more sophisticated)
            generated_code = self._apply_basic_fixes(generated_code, validation_result["errors"])
        
        formatted_code = self.formatter.format_code(generated_code, file_path)
        
        return GeneratedCode(
            code=formatted_code,
            context_used=context,
            validation_result=validation_result
        )
    
    def _generate_with_context_intent_and_patterns(self, context: Context, request: str, 
                                                   intent: IntentType, pattern_analysis: Dict) -> str:
        """Internal method to generate code with context, intent, and patterns"""
        # Format context for the LLM
        formatted_context = format_context_for_llm(context)
        
        # Get relevant templates
        relevant_templates = pattern_analysis.get('relevant_templates', [])
        
        # Build generation prompt adapted to intent
        system_prompt = _build_system_prompt_for_intent(intent)
        
        # Include templates in the user prompt if available
        templates_text = ""
        if relevant_templates:
            templates_text = "\n\nConsider these relevant templates from the codebase:\n" + "\n\n".join(relevant_templates[:2])  # Limit to 2 templates
        
        user_prompt = _build_user_prompt_for_intent(
            request, 
            formatted_context + templates_text, 
            "unknown.py",  # Will be replaced in the actual call
            0,  # Will be replaced in the actual call
            intent
        )
        
        # Generate code using LocalLLM
        generated_code = self.generator.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=512,
            context_chunks=[formatted_context] if formatted_context else None
        )
        
        return generated_code
    
    def _generate_with_context(self, context: Context, request: str) -> str:
        """Internal method to generate code with context"""
        # Format context for the LLM
        formatted_context = format_context_for_llm(context)
        
        # Build generation prompt
        system_prompt = (
            "You are an expert code assistant. Generate code based on the user's request "
            "while maintaining consistency with the existing codebase. "
            "Follow the patterns and conventions shown in the context."
        )
        
        user_prompt = (
            f"Request: {request}\n\n"
            f"Context:\n{formatted_context}\n\n"
            "Generate appropriate code:"
        )
        
        # Generate code using LocalLLM
        generated_code = self.generator.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=512,
            context_chunks=[formatted_context] if formatted_context else None
        )
        
        return generated_code
    
    def _apply_basic_fixes(self, code: str, errors: List[Dict]) -> str:
        """Apply basic fixes to common syntax errors"""
        # This is a simplified implementation
        # In a real implementation, this would be more sophisticated
        fixed_code = code
        
        for error in errors:
            if error["type"] == "syntax_error":
                # Attempt basic fix for common Python syntax errors
                if "invalid syntax" in error["message"]:
                    # Could implement specific fixes based on error details
                    pass
        
        return fixed_code