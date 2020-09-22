__version__ = '0.1.0'

import os, sys
if 'SGE_CELL' in os.environ:
    print(f"Running on {os.environ['HOSTNAME']}.", file=sys.stdout)
