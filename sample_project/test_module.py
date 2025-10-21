def hello_world(name: str) -> str:
    """
    A simple function that returns a greeting message.
    
    Args:
        name: The name of the person to greet
        
    Returns:
        A greeting message with the person's name
    """
    return f"Hello, {name}! Welcome to semindex AI features testing."


class Calculator:
    """A simple calculator class for testing purposes."""
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract the second number from the first."""
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """Divide the first number by the second."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b