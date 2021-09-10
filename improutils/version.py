__version__ = "1.1.7"

if os.environ.get('TARGET_ENV'):
    __version__ = __version__ + "-" + os.environ['CI_JOB_ID']