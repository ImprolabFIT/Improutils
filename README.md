This repository contains Python library with useful helper functions for Machine vision and Image processing (BI-SVZ) coursework taught at [FIT CTU](https://fit.cvut.cz/en). For more details, see [main course page](https://github.com/ImprolabFIT/BI-SVZ-coursework).

### How to deploy a new version
This repository uses GitLab CI/CD Pipelines to deploy improutils package either to production or test PyPI.

#### Steps to release a new version to production PyPI
 - Commit your local changes
	 - ```git commit -m "Add awesome AI feature" ```
 - Create a tag version with respect to [semantic versioning rules](https://semver.org/)
	 - ```git tag 'x.x.x'```
 - Push tag to origin
	 - ```git push --tags```
 - Wait for the package to be deployed and then check new version at [PyPI](https://pypi.org/project/improutils/)

#### Steps to release a new version to test PyPI
Almost same as above, but the **create and push tag step** must be skipped. Testing version is available at [test PyPI](https://test.pypi.org/project/improutils/).
