import json
import datetime
from pathlib import Path
from config import get_constant
from icecream import ic


def aggregate_file_states() -> str:
    """
    Aggregate file states. Here we reuse extract_files_content.
    """
    return extract_files_content()

def extract_files_content() -> str:
    """ Returns the complete contents of all files with special handling for images """
    LOG_FILE = Path(get_constant('LOG_FILE'))
    # Create the file if it does not exist
    if not LOG_FILE.exists():
        LOG_FILE.touch()
    # Read the log file
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        # Handle case of empty file
        if f.read() == "":
            return "No code yet"
        f.seek(0)  # Reset file pointer to the beginning
        logs = json.loads(f.read())
    
    # Initialize output string
    output = []
    
    # Process each file in the logs
    for filepath in logs.keys():
        try:
            # Convert Windows path to Path object
            path = Path(filepath)
            
            # Skip if file doesn't exist
            if not path.exists():
                continue
            
            # Add file header
            output.append(f"# filepath: {filepath}")
            
            # Check if it's an image file
            if path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                output.append("[Image File]")
                output.append(f"Created: {logs[filepath]['created_at']}")
                output.append(f"Last Operation: {logs[filepath]['operations'][-1]['operation']}")
            else:
                # For non-image files, include the content
                content = path.read_text(encoding='utf-8')
                output.append(content)
            
            output.append("\n" + "=" * 80 + "\n")  # Separator between files
            
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            continue
    
    # Combine all content
    return "\n".join(output)


def log_file_operation(path: Path, operation: str) -> None:
    """Log operations on a file with timestamp.
    
    Args:
        path (Path): Path to the file being operated on
        operation (str): Type of operation (e.g., 'create', 'modify', 'delete')
    """
    try:
        LOG_FILE = Path(get_constant('LOG_FILE'))
        
        # Initialize default log structure
        logs = {}
        
        # Read existing logs if file exists and has content
        if LOG_FILE.exists():
            content = LOG_FILE.read_text(encoding='utf-8').strip()
            if content:
                try:
                    logs = json.loads(content)
                except json.JSONDecodeError:
                    logs = {}
        
        path_str = str(path)
        
        # Create new entry if file not logged before
        if path_str not in logs:
            logs[path_str] = {
                "created_at": datetime.datetime.now().isoformat(),
                "operations": []
            }
        
        # Add new operation
        logs[path_str]["operations"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "operation": operation
        })
        
        # Write updated logs
        LOG_FILE.write_text(json.dumps(logs, indent=2), encoding='utf-8')
        
    except Exception as e:
        ic(f"Failed to log file operation: {str(e)}")

def get_all_current_code() -> str:
    """ Returns the complete contents of all files with special handling for images """
    try:
        LOG_FILE = Path(get_constant('LOG_FILE'))
        # Create the file if it does not exist
        if not LOG_FILE.exists():
            LOG_FILE.touch()
        # Read the log file
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            # Handle case of empty file
            if f.read() == "":
                return "No code yet"
            f.seek(0)  # Reset file pointer to the beginning
            logs = json.loads(f.read())
        
        # Initialize output string
        output = []
        
        # Process each file in the logs
        for filepath in logs.keys():
            try:
                # Convert Windows path to Path object
                path = Path(filepath)
                
                # Skip if file doesn't exist
                if not path.exists():
                    continue
                
                # Add file header
                output.append(f"# filepath: {filepath}")
                
                # Check if it's an image file
                if path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                    output.append("[Image File]")
                    output.append(f"Created: {logs[filepath]['created_at']}")
                    output.append(f"Last Operation: {logs[filepath]['operations'][-1]['operation']}")
                else:
                    # For non-image files, include the content
                    content = path.read_text(encoding='utf-8')
                    output.append(content)
                
                output.append("\n" + "=" * 80 + "\n")  # Separator between files
                
            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")
                continue
        
        # Combine all content
        return "\n".join(output)
        
    except Exception as e:
        return f"Error reading log file: {str(e)}"