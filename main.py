# Python 3.15 Workspace
# This is a new Python workspace created for your project.

import os
import sys

print("Hello, Python 3.15!")
print("Current Directory (Direct):", os.getcwd())
print("Current Directory (Indirect):", 
	  os.path.dirname(os.path.abspath(__file__)))
print("Python Version:", sys.version)