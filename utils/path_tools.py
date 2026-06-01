import os

def get_project_root():
    """Module docstring."""
    # (encoding fixed)
    current_file = os.path.abspath(__file__)
    # (encoding fixed)
    current_dir = os.path.dirname(current_file)
    # (encoding fixed)
    project_root = os.path.dirname(current_dir)
    # (encoding fixed)
    return project_root

def get_abs_path(relative_path):
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)


if __name__ == '__main__':
    print(get_abs_path("utils/path_tools.py"))
