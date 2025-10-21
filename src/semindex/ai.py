"""
AI Command Module for semindex
Provides advanced AI capabilities for code understanding, generation and analysis
"""
import argparse
import json
import os
import sqlite3
from typing import List, Dict, Any, Optional, Tuple, Callable, Union

from .rag import retrieve_context, generate_answer
from .local_llm import LocalLLM
from .embed import Embedder
from .model import Symbol
from .store import DB_NAME


class ConversationMemory:
    """Manages conversation history for AI chat sessions."""
    
    def __init__(self, max_entries: int = 10):
        self.max_entries = max_entries
        self.history: List[Dict[str, str]] = []
    
    def add_interaction(self, user_input: str, ai_response: str) -> None:
        """Add a user-AI interaction to the conversation history."""
        self.history.append({
            "user": user_input,
            "ai": ai_response
        })
        
        # Keep only the most recent entries
        if len(self.history) > self.max_entries:
            self.history = self.history[-self.max_entries:]
    
    def get_context(self) -> str:
        """Get the conversation context as a formatted string."""
        if not self.history:
            return ""
        
        context_parts = ["Previous conversation:"]
        for i, entry in enumerate(self.history[-5:], 1):  # Include last 5 exchanges
            context_parts.append(f"Q{i}: {entry['user']}")
            context_parts.append(f"A{i}: {entry['ai']}")
            context_parts.append("---")
        
        return "\n".join(context_parts)


class FunctionCallExecutor:
    """Executes function calls for code analysis tasks."""
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.db_path = os.path.join(index_dir, DB_NAME)
        self.functions = {
            "search_code": self.search_code,
            "explain_code": self.explain_code,
            "find_bugs": self.find_bugs,
            "suggest_refactoring": self.suggest_refactoring,
            "generate_tests": self.generate_tests,
            "get_symbol_info": self.get_symbol_info,
        }
    
    def search_code(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the codebase semantically based on the query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching code snippets with metadata
        """
        try:
            from .rag import retrieve_context
            results = retrieve_context(
                index_dir=self.index_dir,
                query=query,
                top_k=top_k,
                hybrid=False  # Using semantic search by default
            )
            
            # Process results and return structured data
            processed_results = []
            for score, snippet in results:
                # Extract file path and lines from snippet header
                lines = snippet.split('\n')
                header = lines[0] if lines else ""
                
                result = {
                    "score": float(score),
                    "snippet": "\n".join(lines[1:]),  # Exclude header from snippet
                    "header": header,
                    "path": "",  # Will extract from header
                    "lines": ""    # Will extract from header
                }
                
                # Parse header to extract path and line numbers
                if header.startswith("# File: "):
                    # Format is "# File: path:start-end"
                    path_section = header[8:]  # Remove "# File: "
                    if ':' in path_section:
                        path, lines = path_section.rsplit(':', 1)
                        result["path"] = path
                        result["lines"] = lines
                
                processed_results.append(result)
            
            return processed_results
        except Exception as e:
            return [{"error": f"Error searching code: {str(e)}"}]
    
    def explain_code(self, code: str) -> str:
        """
        Generate an explanation for the given code snippet.
        
        Args:
            code: Code snippet to explain
            
        Returns:
            Explanation of the code
        """
        try:
            # Use the existing RAG system to generate an explanation
            llm = LocalLLM(model_type="transformer")
            system_prompt = (
                "You are a code explanation assistant. Provide clear, concise explanations "
                "of code functionality. Focus on what the code does, not how to improve it."
            )
            
            user_prompt = f"Explain the following code:\n\n{code}"
            response = llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=512
            )
            
            return response
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def find_bugs(self, code: str) -> List[Dict[str, str]]:
        """
        Analyze code for potential bugs.
        
        Args:
            code: Code to analyze for bugs
            
        Returns:
            List of potential bugs found
        """
        try:
            llm = LocalLLM(model_type="transformer")
            system_prompt = (
                "You are a code bug detection assistant. Identify potential bugs, security vulnerabilities, "
                "and logical errors in the provided code. Be specific about the issues found."
            )
            
            user_prompt = f"Analyze this code for bugs and vulnerabilities:\n\n{code}"
            response = llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=512
            )
            
            # In a real implementation, we would parse the response into structured bug reports
            # For now, we'll return the raw analysis
            return [{"analysis": response}]
        except Exception as e:
            return [{"error": f"Error analyzing code for bugs: {str(e)}"}]
    
    def suggest_refactoring(self, code: str) -> List[Dict[str, str]]:
        """
        Suggest refactoring opportunities for the given code.
        
        Args:
            code: Code to analyze for refactoring opportunities
            
        Returns:
            List of refactoring suggestions
        """
        try:
            llm = LocalLLM(model_type="transformer")
            system_prompt = (
                "You are a code refactoring assistant. Suggest specific improvements to the code "
                "to enhance readability, performance, maintainability, and best practices. "
                "Provide concrete suggestions."
            )
            
            user_prompt = f"Suggest refactoring for this code:\n\n{code}"
            response = llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=512
            )
            
            return [{"suggestion": response}]
        except Exception as e:
            return [{"error": f"Error suggesting refactoring: {str(e)}"}]
    
    def generate_tests(self, code: str, framework: str = "pytest") -> str:
        """
        Generate unit tests for the given code.
        
        Args:
            code: Code to generate tests for
            framework: Testing framework to use (default: pytest)
            
        Returns:
            Generated test code
        """
        try:
            llm = LocalLLM(model_type="transformer")
            system_prompt = (
                f"You are a unit test generation assistant. Generate comprehensive unit tests using {framework} "
                "for the provided code. Include tests for normal cases, edge cases, and error conditions."
            )
            
            user_prompt = f"Generate {framework} tests for this code:\n\n{code}"
            response = llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1024
            )
            
            return response
        except Exception as e:
            return f"Error generating tests: {str(e)}"
    
    def get_symbol_info(self, symbol_name: str) -> List[Dict[str, Any]]:
        """
        Get information about a specific symbol from the index.
        
        Args:
            symbol_name: Name of the symbol to look up
            
        Returns:
            List of symbol information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query the symbols table for the given name
            cursor.execute(
                "SELECT path, name, kind, start_line, end_line, signature, docstring FROM symbols WHERE name = ?",
                (symbol_name,)
            )
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                results.append({
                    "path": row[0],
                    "name": row[1],
                    "kind": row[2],  # e.g. function, class, module
                    "start_line": row[3],
                    "end_line": row[4],
                    "signature": row[5],
                    "docstring": row[6]
                })
            
            conn.close()
            return results
        except Exception as e:
            return [{"error": f"Error getting symbol info: {str(e)}"}]
    
    def execute(self, function_name: str, **kwargs) -> Any:
        """
        Execute a specific function with the given arguments.
        
        Args:
            function_name: Name of the function to execute
            **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function execution
        """
        if function_name not in self.functions:
            return {"error": f"Function '{function_name}' not available"}
        
        func = self.functions[function_name]
        try:
            return func(**kwargs)
        except Exception as e:
            return {"error": f"Error executing {function_name}: {str(e)}"}


def cmd_ai_chat(args: argparse.Namespace) -> None:
    """Start an interactive AI chat session about the codebase."""
    print(f"Starting AI chat session with model: {args.model}")
    print("Type 'exit' or 'quit' to end the session.")
    print("-" * 50)
    
    # Initialize the LLM
    llm = LocalLLM(
        model_type="transformer",
        model_name=os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    )
    
    # Initialize conversation memory
    memory = ConversationMemory(max_entries=10)
    
    # Initialize function executor
    func_executor = FunctionCallExecutor(index_dir=args.index_dir)
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Ending AI chat session.")
            break
            
        if not user_input:
            continue
            
        # Check if user wants to execute a specific function
        if user_input.startswith("!"):
            # Parse function call: !function_name arg1 arg2 ...
            parts = user_input[1:].split(" ", 1)  # Remove '!' and split into function and args
            if len(parts) >= 1:
                func_name = parts[0]
                func_args = parts[1] if len(parts) > 1 else ""
                
                # Execute the function
                if func_name == "search":
                    results = func_executor.search_code(query=func_args or "all", top_k=args.top_k)
                    print("\nAI: Search Results:")
                    for i, result in enumerate(results[:args.top_k], 1):
                        if "error" not in result:
                            print(f"  {i}. {result.get('path', 'Unknown')} - {result.get('header', 'No header')}")
                            print(f"     Snippet: {result.get('snippet', '')[:200]}...")
                        else:
                            print(f"  Error: {result['error']}")
                    continue
                elif func_name == "symbol":
                    results = func_executor.get_symbol_info(symbol_name=func_args)
                    print(f"\nAI: Symbol Information for '{func_args}':")
                    for result in results:
                        if "error" not in result:
                            print(f"  Path: {result.get('path', 'Unknown')}")
                            print(f"  Kind: {result.get('kind', 'Unknown')}")
                            print(f"  Lines: {result.get('start_line', '?')}-{result.get('end_line', '?')}")
                            if result.get('signature'):
                                print(f"  Signature: {result.get('signature')}")
                            if result.get('docstring'):
                                print(f"  Docstring: {result.get('docstring')}")
                        else:
                            print(f"  Error: {result['error']}")
                    continue
        
        # For regular chat, retrieve relevant context from the codebase
        context_snippets = []
        if args.index_dir and os.path.exists(args.index_dir):
            try:
                snippets = retrieve_context(
                    index_dir=args.index_dir,
                    query=user_input,
                    top_k=args.top_k,
                    hybrid=args.hybrid
                )
                # Take top snippets sorted by relevance
                context_snippets = [s for _, s in sorted(snippets, key=lambda x: x[0], reverse=True)]
            except Exception as e:
                print(f"[WARN] Could not retrieve context: {e}")
        
        # Build comprehensive prompt with conversation history
        system_prompt = (
            "You are a local code assistant. Answer accurately using the provided code/documentation snippets. "
            "Cite file paths inline where relevant. If uncertain, say so. Be helpful but concise."
        )
        
        # Get conversation context
        conversation_context = memory.get_context()
        
        # Combine all context
        all_context = []
        if context_snippets:
            all_context.extend(context_snippets)
        if conversation_context:
            all_context.append(conversation_context)
        
        # Generate response
        try:
            response = llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_input,
                context_chunks=all_context if all_context else None,
                max_tokens=args.max_tokens
            )
            print(f"\nAI: {response}")
            
            # Add to conversation memory
            memory.add_interaction(user_input, response)
        except Exception as e:
            print(f"\nError generating response: {e}")


def cmd_ai_explain(args: argparse.Namespace) -> None:
    """Explain code functionality based on user query."""
    if not args.target:
        print("Error: Please specify what code to explain")
        return
    
    # Retrieve relevant context from the codebase
    context_snippets = retrieve_context(
        index_dir=args.index_dir,
        query=f"Explain the functionality of {args.target}",
        top_k=args.top_k,
        hybrid=args.hybrid
    )
    
    # Sort by relevance
    sorted_snippets = [s for _, s in sorted(context_snippets, key=lambda x: x[0], reverse=True)]
    
    if not sorted_snippets:
        print(f"No relevant code found for: {args.target}")
        return
    
    # Prepare system prompt for explanation
    system_prompt = (
        "You are a code explanation assistant. Provide clear, concise explanations "
        "of code functionality based on the provided context snippets. "
        "Include relevant file paths and line numbers when explaining. "
        "Focus on what the code does, not how to improve it."
    )
    
    # Generate explanation
    user_prompt = f"Explain the functionality of {args.target}"
    llm = LocalLLM(model_path=args.llm_path) if args.llm_path else LocalLLM()
    
    response = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_chunks=sorted_snippets,
        max_tokens=args.max_tokens
    )
    
    print(f"Explanation of {args.target}:")
    print("-" * 40)
    print(response)


def cmd_ai_suggest(args: argparse.Namespace) -> None:
    """Suggest improvements to the codebase."""
    # Retrieve relevant context from the codebase
    context_snippets = retrieve_context(
        index_dir=args.index_dir,
        query="code quality issues, optimization opportunities, best practices",
        top_k=args.top_k,
        hybrid=args.hybrid
    )
    
    # Sort by relevance
    sorted_snippets = [s for _, s in sorted(context_snippets, key=lambda x: x[0], reverse=True)]
    
    if not sorted_snippets:
        print("No code found to analyze for suggestions")
        return
    
    # Prepare system prompt for suggestions
    system_prompt = (
        "You are a code improvement assistant. Analyze the provided code snippets "
        "and suggest specific improvements focusing on: performance, readability, "
        "security, maintainability, and adherence to best practices. "
        "Be specific and provide actionable suggestions. "
        "Include relevant file paths and line numbers in your suggestions."
    )
    
    # Generate suggestions
    user_prompt = "Analyze these code snippets and suggest specific improvements"
    llm = LocalLLM(model_path=args.llm_path) if args.llm_path else LocalLLM()
    
    response = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_chunks=sorted_snippets,
        max_tokens=args.max_tokens
    )
    
    print("Code Improvement Suggestions:")
    print("-" * 40)
    print(response)


def cmd_ai_generate(args: argparse.Namespace) -> None:
    """Generate code based on user description."""
    if not args.description:
        print("Error: Please provide a description of the code to generate")
        return
    
    # For code generation, we might want to look for similar patterns in the codebase
    context_snippets = []
    if args.include_context:
        context_snippets = retrieve_context(
            index_dir=args.index_dir,
            query=args.description,
            top_k=args.top_k,
            hybrid=args.hybrid
        )
        context_snippets = [s for _, s in sorted(context_snippets, key=lambda x: x[0], reverse=True)]
    
    # Prepare system prompt for code generation
    system_prompt = (
        "You are a code generation assistant. Generate clean, efficient, and well-documented "
        "code based on the user's description. Follow the patterns and conventions seen in "
        "the provided context if applicable. Make sure the code is complete and functional."
    )
    
    # Generate code
    user_prompt = f"Generate code based on this description: {args.description}"
    llm = LocalLLM(model_path=args.llm_path) if args.llm_path else LocalLLM()
    
    response = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_chunks=context_snippets if context_snippets else None,
        max_tokens=args.max_tokens
    )
    
    print("Generated Code:")
    print("-" * 40)
    print(response)


def cmd_ai_docs(args: argparse.Namespace) -> None:
    """Generate documentation for code elements."""
    if not args.target:
        print("Error: Please specify what to document")
        return
    
    # Retrieve relevant context from the codebase
    context_snippets = retrieve_context(
        index_dir=args.index_dir,
        query=f"code documentation for {args.target}",
        top_k=args.top_k,
        hybrid=args.hybrid
    )
    
    # Sort by relevance
    sorted_snippets = [s for _, s in sorted(context_snippets, key=lambda x: x[0], reverse=True)]
    
    if not sorted_snippets:
        print(f"No relevant code found for: {args.target}")
        return
    
    # Prepare system prompt for documentation
    system_prompt = (
        "You are a code documentation assistant. Generate clear, comprehensive "
        "documentation for the provided code elements. Include purpose, parameters, "
        "return values, exceptions, and usage examples where appropriate. "
        "Follow common documentation standards for the language."
    )
    
    # Generate documentation
    user_prompt = f"Generate documentation for {args.target}"
    llm = LocalLLM(
        model_type="transformer",
        model_name=os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    )
    
    response = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_chunks=sorted_snippets,
        max_tokens=args.max_tokens
    )
    
    print(f"Documentation for {args.target}:")
    print("-" * 40)
    print(response)


def cmd_ai_find_bugs(args: argparse.Namespace) -> None:
    """Find potential bugs in the codebase."""
    # Initialize function executor
    func_executor = FunctionCallExecutor(index_dir=args.index_dir)
    
    # If a specific target is provided, analyze that specific code
    if args.target:
        # Get the code for the specific target
        symbol_info = func_executor.get_symbol_info(symbol_name=args.target)
        if symbol_info and "error" not in symbol_info[0]:
            code_snippet = f"File: {symbol_info[0]['path']}\n"
            code_snippet += f"Function: {symbol_info[0]['name']}\n"
            code_snippet += f"Signature: {symbol_info[0]['signature']}\n"
            code_snippet += f"Kind: {symbol_info[0]['kind']}\n"
            
            # Load the actual code from the file
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
            
            bugs = func_executor.find_bugs(code=code_snippet)
            print(f"Potential bugs found in {args.target}:")
            print("-" * 40)
            for bug in bugs:
                if "error" in bug:
                    print(f"Error: {bug['error']}")
                else:
                    print(bug.get("analysis", "No analysis provided"))
        else:
            print(f"Could not find symbol: {args.target}")
    else:
        # If no specific target, search for code that might have bugs
        print("Searching for potentially problematic code patterns...")
        bug_indicators = [
            "error handling",
            "exception",
            "try catch",
            "null pointer",
            "memory leak",
            "race condition",
            "insecure",
            "vulnerability",
            "TODO",
            "FIXME",
            "HACK"
        ]
        
        for indicator in bug_indicators:
            results = func_executor.search_code(query=indicator, top_k=2)
            if results and "error" not in results[0]:
                print(f"\nFound code related to '{indicator}':")
                for result in results:
                    if "error" not in result:
                        print(f"  File: {result.get('path', 'Unknown')}")
                        print(f"  Lines: {result.get('lines', '?')}")
                        print(f"  Snippet: {result.get('snippet', '')[:200]}...")
                        print()


def cmd_ai_refactor(args: argparse.Namespace) -> None:
    """Suggest refactoring opportunities in the codebase."""
    # Initialize function executor
    func_executor = FunctionCallExecutor(index_dir=args.index_dir)
    
    # If a specific target is provided, analyze that specific code
    if args.target:
        # Get the code for the specific target
        symbol_info = func_executor.get_symbol_info(symbol_name=args.target)
        if symbol_info and "error" not in symbol_info[0]:
            code_snippet = f"File: {symbol_info[0]['path']}\n"
            code_snippet += f"Function: {symbol_info[0]['name']}\n"
            code_snippet += f"Signature: {symbol_info[0]['signature']}\n"
            code_snippet += f"Kind: {symbol_info[0]['kind']}\n"
            
            # Load the actual code from the file
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
            
            suggestions = func_executor.suggest_refactoring(code=code_snippet)
            print(f"Refactoring suggestions for {args.target}:")
            print("-" * 40)
            for suggestion in suggestions:
                if "error" in suggestion:
                    print(f"Error: {suggestion['error']}")
                else:
                    print(suggestion.get("suggestion", "No suggestion provided"))
        else:
            print(f"Could not find symbol: {args.target}")
    else:
        # If no specific target, search for common refactoring candidates
        print("Looking for refactoring opportunities...")
        refactoring_indicators = [
            "duplicate code",
            "long method",
            "complex",
            "nested if",
            "too many parameters",
            "large class",
            "magic number"
        ]
        
        for indicator in refactoring_indicators:
            results = func_executor.search_code(query=indicator, top_k=2)
            if results and "error" not in results[0]:
                print(f"\nFound code related to '{indicator}':")
                for result in results:
                    if "error" not in result:
                        print(f"  File: {result.get('path', 'Unknown')}")
                        print(f"  Lines: {result.get('lines', '?')}")
                        print(f"  Snippet: {result.get('snippet', '')[:200]}...")
                        print()


def cmd_ai_tests(args: argparse.Namespace) -> None:
    """Generate unit tests for code elements."""
    if not args.target:
        print("Error: Please specify what to create tests for")
        return
    
    # Initialize function executor
    func_executor = FunctionCallExecutor(index_dir=args.index_dir)
    
    # Get the code for the specific target
    symbol_info = func_executor.get_symbol_info(symbol_name=args.target)
    if symbol_info and "error" not in symbol_info[0]:
        code_snippet = f"File: {symbol_info[0]['path']}\n"
        code_snippet += f"Function: {symbol_info[0]['name']}\n"
        code_snippet += f"Signature: {symbol_info[0]['signature']}\n"
        code_snippet += f"Kind: {symbol_info[0]['kind']}\n"
        
        # Load the actual code from the file
        try:
            with open(symbol_info[0]['path'], 'r', encoding='utf-8') as f:
                source = f.read()
                start_line = symbol_info[0]['start_line']
                end_line = symbol_info[0]['end_line']
                lines = source.split('\n')
                code_content = '\n'.join(lines[start_line-1:end_line])
                code_snippet += f"\nCode:\n{code_content}"
        except Exception as e:
            print(f"Error reading file: {e}")
            return
        
        # Generate tests
        test_code = func_executor.generate_tests(
            code=code_snippet,
            framework=args.framework or "pytest"
        )
        
        print(f"Generated tests for {args.target}:")
        print("-" * 40)
        print(test_code)
    else:
        print(f"Could not find symbol: {args.target}")