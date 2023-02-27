# Installing developer (editable) version on windows
This page explains, how to install improutils (and set them up),
in such a way that you can develop the code, and run the tests on your machine.

The basic problem is, that if the repository would be only cloned,
and you would run the tests, then the tests would import the improutils libraries,
which would be tested, resulting in situation, when the downlaoded and edited improutils source files **would not be tested, as the test would run above the installed libraries**.

This page explains, how to deal with this, using virtual environments.

### Requirements
Here are the expected requirements for this to make it work:
 - Windows OS (tested on Windows 11)
 - anaconda prompt

### Instalation
 - open anaconda prompt
 - navigate to the directory with improutils
 - create a virtual environment (if not done already) `conda create -n myenv`
 - activate the virtual environment `activate myenv`
 - install the needed dependencies `pip install -e .`
 - while in the directory with the sources you are developing, run the tests by `python tests\test_acquisition.py`

### Troubleshooting
The following section is only related to various problems, that you may encounter,
during the instalation described in this page.

##### No module named 'numpy.core._multiarray_umath'
<details>
  Here is an example of the original error:
  <summary>Click to expand!</summary>

```
(myenv) C:\Users\XXX\Desktop\git\improutils_package>python tests\test_acquisition.py
Traceback (most recent call last):
  File "C:\Program Files (x86)\Spyder\Python\Lib\site-packages\numpy\core\__init__.py", line 22, in <module>
    from . import multiarray
  File "C:\Program Files (x86)\Spyder\Python\Lib\site-packages\numpy\core\multiarray.py", line 12, in <module>
    from . import overrides
  File "C:\Program Files (x86)\Spyder\Python\Lib\site-packages\numpy\core\overrides.py", line 7, in <module>
    from numpy.core._multiarray_umath import (
ModuleNotFoundError: No module named 'numpy.core._multiarray_umath'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\XXX\Desktop\git\improutils_package\tests\test_acquisition.py", line 2, in <module>
    from improutils.visualisation import *
  File "C:\Users\XXX\Desktop\git\improutils_package\improutils\__init__.py", line 1, in <module>
    from .other import *
  File "C:\Users\XXX\Desktop\git\improutils_package\improutils\other.py", line 1, in <module>
    import numpy as np
  File "C:\Program Files (x86)\Spyder\Python\Lib\site-packages\numpy\__init__.py", line 150, in <module>
    from . import core
  File "C:\Program Files (x86)\Spyder\Python\Lib\site-packages\numpy\core\__init__.py", line 48, in <module>
    raise ImportError(msg)
ImportError:

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python3.9 from "C:\ProgramFiles\Anaconda3\envs\myenv\python.exe"
  * The NumPy version is: "1.21.2"

and make sure that they are the versions you expect.
Please carefully study the documentation linked above for further help.

Original error was: No module named 'numpy.core._multiarray_umath'
```

</details>


###### solution
 - `pip uninstall numpy`
 - `conda install numpy`

##### No module named 'cv2.cv2'
 - `pip install opencv-python`
 - `pip uninstall opencv-python`


##### pytesseract.pytesseract.TesseractNotFoundError: tesseract is not installed or it's not in your PATH. See README file for more information.
This error may raise, when running the `python tests\test_recognition.py` test

###### solution
Install the teseract ocr by
 - `pip install tesseract-ocr`

This may fail with error due to missing C++ libraries.
The error then looks like this:
```
error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
```
