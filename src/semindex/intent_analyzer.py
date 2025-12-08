"""
Intent Analysis Module for semindex
Provides intent classification and task decomposition for code generation requests.
"""
import os
import re
from typing import Dict, List, Tuple, Optional
from enum import Enum

from .local_llm import LocalLLM


class IntentType(Enum):
    IMPLEMENTATION = "implementation"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    ANALYSIS = "analysis"


class IntentClassifier:
    """Classifies user intent for code generation requests"""
    
    def __init__(self, index_dir: str = None):
        self.index_dir = index_dir
        self.llm = LocalLLM(
            model_type="transformer",
            model_name=os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        )
    
    def classify_intent(self, user_request: str) -> Dict:
        """
        Classify the user's request intent using pattern matching and LLM analysis
        """
        # First, try pattern-based classification
        intent = self._pattern_based_classification(user_request)
        
        # If pattern matching is inconclusive, use LLM
        if intent == IntentType.ANALYSIS:  # Default fallback
            intent = self._llm_based_classification(user_request)
        
        # Extract subtasks if possible
        subtasks = self._decompose_request(user_request, intent)
        
        return {
            'intent': intent,
            'confidence': 0.8,  # Placeholder confidence
            'subtasks': subtasks,
            'original_request': user_request
        }
    
    def _pattern_based_classification(self, request: str) -> IntentType:
        """Classify intent based on keyword patterns"""
        request_lower = request.lower()
        
        # Implementation patterns
        implementation_keywords = [
            'implement', 'create', 'add', 'write', 'build', 'develop', 'make',
            'function to', 'class to', 'method to', 'code to', 'new'
        ]
        
        # Refactoring patterns
        refactoring_keywords = [
            'refactor', 'improve', 'optimize', 'simplify', 'clean up', 'restructure',
            'better way', 'more efficient', 'upgrade', 'modernize', 'reorganize'
        ]
        
        # Debugging patterns
        debugging_keywords = [
            'fix', 'debug', 'error', 'bug', 'issue', 'problem', 'not working',
            'troubleshoot', 'crash', 'exception', 'traceback', 'trace back'
        ]
        
        # Documentation patterns
        documentation_keywords = [
            'document', 'docstring', 'comment', 'explain', 'description',
            'what does', 'how does', 'purpose of', 'describe', 'summarize'
        ]
        
        # Testing patterns
        testing_keywords = [
            'test', 'unittest', 'pytest', 'check if', 'verify', 'assert',
            'validation', 'ensure that', 'confirm that', 'validate'
        ]
        
        # Count keyword matches for each category
        impl_count = sum(1 for keyword in implementation_keywords if keyword in request_lower)
        ref_count = sum(1 for keyword in refactoring_keywords if keyword in request_lower)
        debug_count = sum(1 for keyword in debugging_keywords if keyword in request_lower)
        doc_count = sum(1 for keyword in documentation_keywords if keyword in request_lower)
        test_count = sum(1 for keyword in testing_keywords if keyword in request_lower)
        
        # Determine intent based on highest count
        counts = [
            (impl_count, IntentType.IMPLEMENTATION),
            (ref_count, IntentType.REFACTORING),
            (debug_count, IntentType.DEBUGGING),
            (doc_count, IntentType.DOCUMENTATION),
            (test_count, IntentType.TESTING)
        ]
        
        max_count, max_intent = max(counts, key=lambda x: x[0])
        
        # If no strong indicator, return analysis as default
        if max_count == 0:
            return IntentType.ANALYSIS
        
        return max_intent
    
    def _llm_based_classification(self, request: str) -> IntentType:
        """Use LLM to classify intent when pattern matching is unclear"""
        system_prompt = """
        You are an intent classifier for code generation requests. 
        Categorize the user's request into one of these types:
        - IMPLEMENTATION: New code implementation
        - REFACTORING: Code improvement or restructuring  
        - DEBUGGING: Bug identification or fixing
        - DOCUMENTATION: Documentation generation
        - TESTING: Test case creation
        - ANALYSIS: Code understanding or explanation
        
        Respond with just the intent type name.
        """
        
        try:
            response = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=f"Request: {request}\n\nIntent type:",
                max_tokens=20
            )
            
            # Parse the response
            response_lower = response.lower().strip()
            
            if 'implementation' in response_lower or 'new' in response_lower:
                return IntentType.IMPLEMENTATION
            elif 'refactor' in response_lower or 'improv' in response_lower:
                return IntentType.REFACTORING
            elif 'debug' in response_lower or 'bug' in response_lower or 'fix' in response_lower:
                return IntentType.DEBUGGING
            elif 'document' in response_lower or 'comment' in response_lower or 'explain' in response_lower:
                return IntentType.DOCUMENTATION
            elif 'test' in response_lower or 'verify' in response_lower:
                return IntentType.TESTING
            else:
                return IntentType.ANALYSIS
        except:
            return IntentType.ANALYSIS  # Default fallback
    
    def _decompose_request(self, request: str, intent: IntentType) -> List[str]:
        """Decompose complex requests into smaller tasks"""
        # For now, return the original request as a single task
        # In the future, this could be enhanced to identify multiple tasks
        return [request]


class TaskDecomposer:
    """Decomposes complex code generation requests into subtasks"""
    
    def __init__(self, index_dir: str = None):
        self.index_dir = index_dir
        self.llm = LocalLLM(
            model_type="transformer",
            model_name=os.environ.get("SEMINDEX_TRANSFORMER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        )
    
    def decompose_task(self, request: str, intent: IntentType) -> List[Dict[str, str]]:
        """
        Decompose a complex request into smaller, actionable tasks
        """
        if intent == IntentType.IMPLEMENTATION:
            return self._decompose_implementation(request)
        elif intent == IntentType.REFACTORING:
            return self._decompose_refactoring(request)
        elif intent == IntentType.DEBUGGING:
            return self._decompose_debugging(request)
        else:
            # For other intents, return the original request as a single task
            return [{
                'description': request,
                'type': intent.value,
                'priority': 1,
                'dependencies': []
            }]
    
    def _decompose_implementation(self, request: str) -> List[Dict[str, str]]:
        """Decompose implementation requests into smaller tasks"""
        # Try to identify if the request contains multiple requirements
        subtasks = []
        
        # Check for multiple requirements in the request
        and_separators = [' and ', ' with ', ' as well as ', ' plus ']
        or_separators = [' or ']
        
        # This is a basic implementation - could be enhanced with NLP
        parts = [request]  # For now, just return the original
        
        for part in parts:
            subtasks.append({
                'description': part.strip(),
                'type': 'implementation',
                'priority': 1,
                'dependencies': []
            })
        
        return subtasks
    
    def _decompose_refactoring(self, request: str) -> List[Dict[str, str]]:
        """Decompose refactoring requests into smaller tasks"""
        return [{
            'description': request,
            'type': 'refactoring',
            'priority': 1,
            'dependencies': []
        }]
    
    def _decompose_debugging(self, request: str) -> List[Dict[str, str]]:
        """Decompose debugging requests into smaller tasks"""
        return [{
            'description': request,
            'type': 'debugging',
            'priority': 1,
            'dependencies': []
        }]


def analyze_intent_and_decompose(user_request: str, index_dir: str = None) -> Dict:
    """Main function to analyze intent and decompose tasks"""
    classifier = IntentClassifier(index_dir)
    result = classifier.classify_intent(user_request)
    
    decomposer = TaskDecomposer(index_dir)
    result['subtasks'] = decomposer.decompose_task(user_request, result['intent'])
    
    return result