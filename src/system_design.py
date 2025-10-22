# Generated component: System_Design_implementation
# Description: Implementation for task: Create system architecture

import re


def create_system_architecture(user_input):
    """
    This function takes in a user input string representing a system architecture. It returns a dictionary of system components and their respective configurations.
    
    Args:
        user_input (str): A string representing a system architecture.
        
    Returns:
        dict: A dictionary containing system components and their respective configurations.
    """
    
    # Validate user input
    if not isinstance(user_input, str):
        raise TypeError("User input must be a string.")
    
    # Split user input into components and configurations
    components = user_input.split(",")
    configurations = {}
    
    # Loop through each component and configuration pair
    for i, component in enumerate(components):
        if "=" not in component:
            raise ValueError("Component must contain an equal sign.")
        elif "," in component:
            raise ValueError("Configuration must not contain commas.")
        else:
            config = component.strip().split("=")
            
            # Check if configuration is valid
            if len(config) != 2:
                raise ValueError("Configuration must have two elements.")
            
            # Add configuration to dictionary
            configurations[config[0].strip()] = config[1].strip()
    
    # Return dictionary of components and configurations
    return configurations