import setuptools
import os        
           
if os.environ.get('CI_COMMIT_TAG'):
    version = os.environ['CI_COMMIT_TAG']
elif os.environ.get('CI_JOB_ID'):
    version = os.environ['CI_JOB_ID'] 
else:
    version = "0.0.0"

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="improutils",
    version=version,
    author="ImproLab",
    author_email="improlab@fit.cvut.cz",
    description="Package with useful functions for BI-SVZ coursework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ImprolabFIT/improutils_package",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
