"""
AI Implementation Assistant for semindex
Provides AI-powered code understanding and implementation suggestions
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .ai import FunctionCallExecutor
from .local_llm import LocalLLM
from .model import Symbol
from .perplexica_adapter import PerplexicaSearchAdapter


class AIImplementationAssistant:
    """AI assistant for understanding code and suggesting implementations"""
    
    def __init__(self, index_dir: str, config_path: str = None):
        self.index_dir = index_dir
        self.function_executor = FunctionCallExecutor(index_dir=index_dir)
        self.perplexica_adapter = PerplexicaSearchAdapter(config_path=config_path)
        
    def analyze_code_structure(self, target: str = "") -> str:
        """Analyze the code structure and return insights"""
        query = f"Analyze the overall code structure and architecture of {target}" if target else "Analyze the overall code structure and architecture"
        
        # Search for relevant architecture patterns
        results = self.function_executor.search_code(query=query, top_k=10)
        
        # Build context from the search results
        context = []
        for result in results:
            if "error" not in result:
                context.append(f"File: {result.get('path', 'Unknown')}\nLines: {result.get('lines', '?')}\n{result.get('snippet', '')}")
        
        # Use the LLM to analyze the code structure
        system_prompt = (
            "You are an expert software architect. Analyze the provided code snippets to understand "
            "the overall structure, architecture patterns, and design principles of the codebase. "
            "Provide insights about the organization, key components, relationships between modules, "
            "and architectural decisions."
        )
        
        user_prompt = f"Code snippets:\n{'='*50}\n" + "\n\n".join(context[:5]) + f"\n\nBased on these code snippets, analyze the code structure and architecture."
        
        llm = LocalLLM(
            model_type="transformer",
            model_name=os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        )
        
        response = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1024
        )
        
        return response
    
    def suggest_implementation(self, requirement: str, context_file: Optional[str] = None) -> str:
        """Suggest an implementation approach for a given requirement"""
        # Get relevant code context based on the requirement
        results = self.function_executor.search_code(query=requirement, top_k=5)
        
        # Build the context with relevant code snippets
        context_parts = []
        if context_file:
            # If a specific file is provided, get its content
            try:
                with open(context_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                    context_parts.append(f"Current file content ({context_file}):\n{source[:1000]}")
            except Exception as e:
                context_parts.append(f"Could not read {context_file}: {str(e)}")
        
        # Add search results to context
        for result in results:
            if "error" not in result:
                context_parts.append(f"Relevant code in {result.get('path', 'Unknown')}:\n{result.get('snippet', '')}")
        
        # Generate implementation suggestion
        system_prompt = (
            "You are a senior software developer. Based on the provided codebase context and "
            "the requirement, suggest a detailed implementation approach. Consider the existing "
            "architecture, coding patterns, and best practices. Provide code snippets if appropriate."
        )
        
        user_prompt = f"Requirement: {requirement}\n\nContext:\n{'='*50}\n" + "\n\n".join(context_parts) + "\n\nSuggest an implementation approach for this requirement."
        
        llm = LocalLLM(
            model_type="transformer",
            model_name=os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        )
        
        response = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1024
        )
        
        return response
    
    def explain_code_component(self, component_name: str) -> str:
        """Explain what a specific code component does"""
        # Get information about the component
        symbol_info = self.function_executor.get_symbol_info(symbol_name=component_name)
        
        if not symbol_info or "error" in symbol_info[0]:
            return f"Could not find information about component: {component_name}"
        
        # Get the actual code content
        code_snippet = f"Component: {symbol_info[0]['name']}\n"
        code_snippet += f"File: {symbol_info[0]['path']}\n"
        code_snippet += f"Type: {symbol_info[0]['kind']}\n"
        if symbol_info[0]['signature']:
            code_snippet += f"Signature: {symbol_info[0]['signature']}\n"
        if symbol_info[0]['docstring']:
            code_snippet += f"Docstring: {symbol_info[0]['docstring']}\n"
        
        try:
            with open(symbol_info[0]['path'], 'r', encoding='utf-8') as f:
                source = f.read()
                start_line = symbol_info[0]['start_line']
                end_line = symbol_info[0]['end_line']
                lines = source.split('\n')
                code_content = '\n'.join(lines[start_line-1:end_line])
                code_snippet += f"\nCode:\n{code_content}"
        except Exception as e:
            code_snippet += f"\nError reading file: {e}"
        
        # Use the LLM to explain the component
        system_prompt = (
            "You are an expert code reviewer. Explain in detail what the provided code component does, "
            "its purpose in the system, how it works, and any important implementation details. "
            "Focus on the function, inputs, outputs, and internal logic."
        )
        
        user_prompt = f"Component details:\n{code_snippet}\n\nExplain this code component."
        
        llm = LocalLLM(
            model_type="transformer",
            model_name=os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        )
        
        response = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1024
        )
        
        return response
    
    def identify_dependencies(self, component_name: str) -> List[str]:
        """Identify dependencies of a specific component"""
        # Get symbol information for the component
        symbol_info = self.function_executor.get_symbol_info(symbol_name=component_name)
        
        if not symbol_info or "error" in symbol_info[0]:
            return []
        
        # Get imports from the file containing the component
        try:
            with open(symbol_info[0]['path'], 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Find import statements in the file
            import_lines = []
            for line in source.split('\n'):
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    import_lines.append(line)
            
            # Extract module names from import statements
            dependencies = []
            for import_line in import_lines:
                if import_line.startswith('import '):
                    modules = import_line[7:].split(',')
                    for module in modules:
                        module = module.strip().split('.')[0]  # Get base module name
                        if module:
                            dependencies.append(module)
                elif import_line.startswith('from '):
                    parts = import_line.split()
                    if len(parts) > 1:
                        module = parts[1].split('.')[0]  # Get base module name
                        if module:
                            dependencies.append(module)
            
            return dependencies
        except Exception as e:
            print(f"Error identifying dependencies for {component_name}: {e}")
            return []
    
    def check_code_quality(self, file_path: str) -> str:
        """Check code quality of a specific file"""
        try:
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Use the LLM to analyze code quality
            system_prompt = (
                "You are a senior code reviewer. Analyze the provided code for quality issues, "
                "best practices, potential bugs, performance concerns, and architectural improvements. "
                "Provide specific suggestions for improvements."
            )
            
            user_prompt = f"Analyze the following code for quality:\n\nFile: {file_path}\n\nCode:\n{source}"
            
            llm = LocalLLM(
                model_type="transformer",
                model_name=os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            )
            
            response = llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1024
            )
            
            return response
        except Exception as e:
            return f"Error checking code quality for {file_path}: {str(e)}"
    
    def generate_function_signature(self, function_description: str, language: str = "python") -> str:
        """Generate a function signature based on a description"""
        # Search for similar functions in the codebase
        results = self.function_executor.search_code(query=function_description, top_k=3)
        
        # Build context with similar functions
        context_parts = []
        for result in results:
            if "error" not in result:
                context_parts.append(f"Similar function in {result.get('path', 'Unknown')}:\n{result.get('snippet', '')}")
        
        # Generate function signature
        system_prompt = (
            f"You are a senior software developer. Generate an appropriate function signature in {language} "
            f"for the given description. Consider the existing codebase patterns if provided. "
            f"Return only the function signature without implementation."
        )
        
        user_prompt = f"Description: {function_description}\n\nContext from codebase:\n{'='*50}\n" + "\n\n".join(context_parts) + f"\n\nFunction signature:"
        
        llm = LocalLLM(
            model_type="transformer",
            model_name=os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        )
        
        response = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=256
        )
        
        return response
    
    def generate_class_structure(self, class_description: str, language: str = "python") -> str:
        """Generate a class structure based on a description"""
        # Search for similar classes in the codebase
        results = self.function_executor.search_code(query=class_description, top_k=3)
        
        # Build context with similar classes
        context_parts = []
        for result in results:
            if "error" not in result:
                context_parts.append(f"Similar class in {result.get('path', 'Unknown')}:\n{result.get('snippet', '')}")
        
        # Generate class structure
        system_prompt = (
            f"You are a senior software developer. Generate an appropriate class structure in {language} "
            f"for the given description. Consider the existing codebase patterns if provided. "
            f"Include appropriate methods, attributes, and design patterns."
        )
        
        user_prompt = f"Description: {class_description}\n\nContext from codebase:\n{'='*50}\n" + "\n\n".join(context_parts) + f"\n\nClass structure:"
        
        llm = LocalLLM(
            model_type="transformer",
            model_name=os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        )
        
        response = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1024
        )
        
        return response
    
    def search_library_documentation(self, library_name: str, query: str) -> str:
        """Search documentation for a specific library using Perplexica API"""
        if not self.perplexica_adapter.is_available():
            return f"Perplexica API is not available. Cannot search documentation for {library_name}."
        
        search_result = self.perplexica_adapter.search_documentation(query, library_name)
        
        if "error" in search_result:
            return f"Error searching documentation for {library_name}: {search_result['error']}"
        
        # Extract relevant content from the search results
        if "results" in search_result:  # Assuming standard response format
            content_parts = []
            for result in search_result["results"][:3]:  # Take top 3 results
                if "pageContent" in result:
                    content_parts.append(f"Title: {result.get('title', 'No Title')}\n{result['pageContent'][:500]}...")
                elif "content" in result:
                    content_parts.append(f"Title: {result.get('title', 'No Title')}\n{result['content'][:500]}...")
            
            return "\n\n".join(content_parts)
        else:
            return f"Could not extract documentation content for {library_name}: {search_result}"
    
    def explain_with_external_knowledge(self, topic: str) -> str:
        """Explain a topic using both internal codebase knowledge and external documentation"""
        # Get internal context
        internal_results = self.function_executor.search_code(query=topic, top_k=5)
        internal_context = []
        for result in internal_results:
            if "error" not in result:
                internal_context.append(f"Code context in {result.get('path', 'Unknown')}:\n{result.get('snippet', '')}")
        
        # Get external knowledge using Perplexica
        external_content = ""
        if self.perplexica_adapter.is_available():
            external_search = self.perplexica_adapter.search_web(topic)
            if "error" not in external_search:
                # Extract relevant content from external search
                if "results" in external_search:
                    for result in external_search["results"][:2]:  # Take top 2 results
                        if "pageContent" in result:
                            external_content += f"External source: {result.get('title', 'No Title')}\n{result['pageContent'][:300]}...\n\n"
                        elif "content" in result:
                            external_content += f"External source: {result.get('title', 'No Title')}\n{result['content'][:300]}...\n\n"
                elif "message" in external_search:  # If the response has a message field
                    external_content = external_search["message"]
        
        # Combine internal and external context
        system_prompt = (
            "You are an expert software engineer. Provide a comprehensive explanation of the topic "
            "using both the internal codebase context and external knowledge. Be accurate and detailed."
        )
        
        context_parts = []
        if internal_context:
            context_parts.append("Internal codebase context:")
            context_parts.extend(internal_context[:2])  # Use top 2 internal results
        if external_content:
            context_parts.append("\nExternal knowledge:")
            context_parts.append(external_content)
        
        user_prompt = f"Topic: {topic}\n\nContext:\n{'='*50}\n" + "\n".join(context_parts) + "\n\nProvide a comprehensive explanation:"
        
        llm = LocalLLM(
            model_type="transformer",
            model_name=os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        )
        
        response = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1024
        )
        
        return response

    def _get_llm_instance(self):
        """Helper to get the LLM instance - used by other components in the system"""
        from .local_llm import LocalLLM
        return LocalLLM(
            model_type="transformer",
            model_name=os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        )