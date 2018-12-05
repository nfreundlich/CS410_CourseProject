import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gflm",
    version="0.0.1",
    author="H. Wilder, N. Freundlich, Santu Karmaker",
    author_email="hwilder3@illinois.edu, norbert4@illinois.edu, karmake2@illinois.edu",
    description="Mine implicit features using a generative feature language model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/gflm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
