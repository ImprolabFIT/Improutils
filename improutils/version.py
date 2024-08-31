__version__ = "1.2.0"

import os

if os.environ.get('TARGET_ENV'):
    __version__ = __version__ + "-" + os.environ['CI_JOB_ID']

print(os.environ.get('TARGET_ENV'))
print(os.environ.get('CI_JOB_ID'))
