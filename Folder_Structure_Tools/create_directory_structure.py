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
MOA-DeepOutputs/
│
├── deepoutputs_engine/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── tracing.py
│   ├── prompts.py
│   ├── reports.py
│   ├── main.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── openrouter.py
│   │   └── mixture.py
│
├── Traces/
│   ├── markdown/
│   └── json/
'''

# Create the directory structure
create_directory_structure(markdown_structure)
print("Directory structure created successfully!")
