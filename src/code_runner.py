import sys
import io
import time
import traceback
import resource
import json
from contextlib import redirect_stdout, redirect_stderr

def run_code(code_str, timeout=5, max_memory_mb=100):
    """
    Execute the provided code string in a controlled environment.
    
    Args:
        code_str: String containing Python code to execute
        timeout: Maximum execution time in seconds
        max_memory_mb: Maximum memory usage allowed in MB
        
    Returns:
        dict: Results including stdout, stderr, execution time, memory usage, and error info
    """
    # Set resource limits
    resource.setrlimit(resource.RLIMIT_AS, (max_memory_mb * 1024 * 1024, max_memory_mb * 1024 * 1024))
    
    # Prepare capture of stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    # Prepare result dictionary
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "exec_time": 0,
        "memory_usage": 0,
        "error_type": None,
        "error_message": None,
        "traceback": None
    }
    
    # Execute the code
    start_time = time.time()
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Create a namespace for execution
            exec_globals = {}
            exec(code_str, exec_globals)
            
        result["success"] = True
    except Exception as e:
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()
    finally:
        # Capture execution metrics
        result["exec_time"] = time.time() - start_time
        result["stdout"] = stdout_capture.getvalue()
        result["stderr"] = stderr_capture.getvalue()
        
        # Get memory usage (this is approximate)
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            result["memory_usage"] = usage.ru_maxrss / 1024  # Convert to MB
        except:
            result["memory_usage"] = -1  # Unable to determine
    
    return result

if __name__ == "__main__":
    # If script is run directly, it expects code as an argument or from stdin
    if len(sys.argv) > 1:
        # Read code from file instead of using filename as code
        file_path = sys.argv[1]
        try:
            with open(file_path, 'r') as f:
                code_to_run = f.read()
        except Exception as e:
            result = {
                "success": False,
                "error_type": type(e).__name__,
                "error_message": f"Failed to read file: {str(e)}"
            }
            print(json.dumps(result))
            sys.exit(1)
    else:
        code_to_run = sys.stdin.read()
    
    result = run_code(code_to_run)
    print(json.dumps(result)) 