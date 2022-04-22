 - add a `CHANGELOG.md`, containing the code changes summary for the last commit,
 which is visible in nice gitlab markdown
 - add a tester for `other.py` functionalities
 - add comments and variable input type definitions (comments) for the functions,
 as well as more exact comments to various functions
 - add `doc` directory, grouping various `.md` files, which will contain additional information about the project, such as installation tutorials, and troubleshooting
 - add `doc/installDevWin.md`, which contains a in-depth tutorial on how to make this project work on windows OS, with respect to the developer (= how to be able to develop this using virtual environment)
 - add link to `doc/installDevWin.md` into file `README.md`
 - `other.py - order_points()`:
    - crashes, when array in used as input even though the
 description of function said, that an array is requested. The function work for numpy array. Changed the misleading description of that function in this regard.
    - the length of points as input is now required to 4 or more, else an exception is raised.
    The reason for this is, that if less than 4 points were input to the older version of the function, it crashed anyway. This can confuse users, as no one forbid them to input 2 points only, for example.
    - multi-dimensional (more than 2) point as well as one dimensional array inputs
    are now rejected as a invalid input.
    This is because the function crashes anyway for input, which contains as
    ,,point" an array of only one number.
    And if multiple numbers are provided (for example `x`, `y`, `z` coordinate per point),
    the function ignores them anyway (=the result is not sorted by the z or other coordinates).
    This also enforces the same dimension for each point.
    Note: understanding, that creating an NP array where not all elements have the same length prints a warning thanks to python by itself, the same size for each element is not validated by the `order_points()` function
 - put `ndarray` as a type to the function description, instead of `numpy.ndarray`
 everywhere is sources, to make it standardized
 - change the docstring style to one style, especially in `coordconversion.py`
 - `coordconversion.py - convert_pt_from_homogenous()`: fix doc, as doc said ndarray is returned, but tuple was returned
 - change docstring from ''' to """
 - fix grammar typos such as `lenght` to `length`
