# Option 1: Using relative import (recommended when running as part of a package)
# from ..bar.bar import print_hello

# Option 2: Alternative - absolute import from src root
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bar import print_hello

print_hello()
