import os
import pathlib

def is_docker_container():
    """
    Check if the current process is running inside a Docker container.
    
    Returns:
        bool: True if running inside a Docker container, False otherwise.
    """
    # Method 1: Check for .dockerenv file
    if pathlib.Path('/.dockerenv').exists():
        return True
    
    # Method 2: Check /proc/1/cgroup for docker-related entries
    try:
        with open('/proc/1/cgroup', 'r') as f:
            content = f.read()
            if 'docker' in content or 'containerd' in content:
                return True
    except (FileNotFoundError, PermissionError):
        pass
    
    # Method 3: Check for container-related environment variables
    container_env_vars = ['DOCKER_CONTAINER', 'CONTAINER', 'KUBERNETES_SERVICE_HOST']
    for env_var in container_env_vars:
        if os.getenv(env_var):
            return True
    
    return False

def is_root_user():
    """
    Check if the current process is running as root user.
    
    Returns:
        bool: True if running as root, False otherwise.
    """
    # Method 1: Check user ID (most reliable on Unix-like systems)
    if hasattr(os, 'getuid'):
        return os.getuid() == 0
    
    # Method 2: Check environment variables (fallback for systems without getuid)
    user_env_vars = ['USER', 'USERNAME']
    for var in user_env_vars:
        user = os.getenv(var, '').lower()
        if user == 'root':
            return True
    
    # Method 3: Check if we can write to system directories (less reliable)
    try:
        test_paths = ['/etc', '/root', '/sys']
        for path in test_paths:
            if os.path.exists(path) and os.access(path, os.W_OK):
                return True
    except (OSError, PermissionError):
        pass
        
    raise RuntimeError("Could not determine if the current process is running as root user")