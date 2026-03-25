import inspect
import sys
import crewai_tools


def list_all_functions(module):
    return [name for name, obj in inspect.getmembers(module) if inspect.isfunction(obj)]


current_module = sys.modules["crewai_tools"]
print("All functions in module:", list_all_functions(current_module))
