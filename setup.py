import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ilu",
    version="0.2.0",
    author="Guilherme Varela",
    author_email="guilhermevarela@hotmail.com",
    description="iLU: Integrative Learning from Urban Data and Situational Context for City Mobility for City Mobility Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/guilhermevarela/ilu",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
