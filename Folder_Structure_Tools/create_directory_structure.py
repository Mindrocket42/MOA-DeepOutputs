import os

def create_directory_structure(structure_text):
    # Split the text into lines and clean them
    lines = [line.rstrip('\n') for line in structure_text.split('\n') if line.strip()]
    
    base_path = r"B:\dev\projects\agent-system"
    current_path = base_path
    path_stack = []

    for line in lines:
        # Count leading spaces to determine indentation level
        indent = len(line) - len(line.lstrip())
        indent_level = indent // 4  # Assuming 4 spaces per indent level
        
        # Clean the line
        clean_line = line.lstrip().replace('│', '').replace('├──', '').replace('└──', '').replace('─', '').strip()
        
        # Skip empty lines or lines with just tree characters
        if not clean_line or all(c in '│├└─' for c in clean_line):
            continue
            
        # Remove comments
        clean_line = clean_line.split('#')[0].strip()
        
        # Adjust path stack based on indentation
        while len(path_stack) > indent_level:
            path_stack.pop()
            
        if not path_stack:
            current_path = base_path
        else:
            current_path = os.path.join(base_path, *path_stack)
            
        # Add current item to path
        path_stack.append(clean_line.rstrip('/'))
        full_path = os.path.join(base_path, *path_stack)
        
        # Create directory or file
        if clean_line.endswith('/'):
            if not os.path.exists(full_path):
                os.makedirs(full_path)
                print(f"Created directory: {full_path}")
        else:
            # Ensure parent directory exists
            parent_dir = os.path.dirname(full_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            
            # Create file
            if not os.path.exists(full_path):
                with open(full_path, 'w') as f:
                    pass
                print(f"Created file: {full_path}")

# Your existing markdown_structure string remains the same
markdown_structure = '''
agent_system/
│
├── README.md
├── requirements.txt
├── setup.py
│
├── config/
│   ├── __init__.py
│   ├── model_config.yaml       # LLM and model parameters
│   └── system_config.yaml      # System-wide settings
│
├── src/
│   ├── __init__.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── main_agent.py
│   │   ├── sub_agent.py
│   │   └── base_agent.py      # Abstract base class
│   │
│   ├── environment/
│   │   ├── __init__.py
│   │   └── environment.py
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── shared_memory.py
│   │   └── memory_manager.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── llm_wrapper.py     # LLM initialization and interface
│   │
│   └── utils/
│       ├── __init__.py
│       ├── async_utils.py
│       ├── logging_utils.py
│       └── task_decomposer.py
│
├── tests/
│   ├── __init__.py
│   ├── test_main_agent.py
│   ├── test_sub_agent.py
│   └── test_environment.py
│
├── examples/
│   ├── basic_usage.py
│   └── advanced_scenarios.py
│
└── logs/
    └── .gitkeep
'''

# Create the directory structure
create_directory_structure(markdown_structure)
print("Directory structure created successfully!")
