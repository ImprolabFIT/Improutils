import setuptools
import os        

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="improutils",
    author="ImproLab",
    version=open("improutils/version.py").readlines()[-1].split()[-1].strip("\"'"),
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
    install_requires=[
        "matplotlib>=3.3.4",
        "numpy>=1.20.1",
        "opencv-python>=4.5.1.48",
        "Pillow>=8.1.0",
        "Pylon>=0.4.4",
        "pytesseract>=0.3.7",
        "wheel",
      ],
    python_requires='>=3.6',
)
