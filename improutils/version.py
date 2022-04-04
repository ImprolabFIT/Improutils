__version__ = "1.1.9"

if os.environ.get('TARGET_ENV'):
    __version__ = __version__ + "-" + os.environ['CI_JOB_ID']

print(os.environ.get('TARGET_ENV'))
print(os.environ.get('CI_JOB_ID'))