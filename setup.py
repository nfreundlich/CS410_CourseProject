import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED = ["spacy==2.0.16",
            "nltk",
            "certifi==2018.10.15",
            "chardet==3.0.4",
            "cymem==2.0.2",
            "cytoolz==0.9.0.1",
            "dill==0.2.8.2",
            "idna==2.7",
            "metapy==0.2.13",
            "msgpack==0.5.6",
            "msgpack-numpy==0.4.3.2",
            "murmurhash==1.0.1",
            "numpy==1.15.4",
            "pandas==0.23.4",
            "parse==1.9.0",
            "parse-type==0.4.2",
            "plac==0.9.6",
            "preshed==2.0.1",
            "pyreadline==2.1",
            "python-dateutil==2.7.5",
            "pytz==2018.7",
            "regex==2018.1.10",
            "requests==2.20.1",
            "scipy==1.1.0",
            "six==1.11.0",
            "thinc==6.12.0",
            "toolz==0.9.0",
            "tqdm==4.28.1",
            "ujson==1.35",
            "urllib3==1.24.1",
            "wrapt==1.10.11",
            ]

setuptools.setup(
    name="feature_mining",
    version="0.0.24",
    author="H. Wilder, N. Freundlich, Santu Karmaker",
    author_email="hwilder3@illinois.edu, norbert4@illinois.edu, karmake2@illinois.edu",
    description="Mine implicit features using a generative feature language model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(exclude=('tests',)),
    package_dir={'feature_mining': './feature_mining'},
    package_data={'feature_mining': ['data/*.final']},
    install_requires=REQUIRED,
    dependency_links=["https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz#en_core_web_sm"], #en_core_web_sm
    data_files=[
                ('./data', ['./feature_mining/data/iPod.final', ]),
                ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

"""
Interesting resources checked for creating setup.py:
https://stackoverflow.com/questions/779495/python-access-data-in-package-subdirectory
https://docs.python.org/3/distutils/setupscript.html#installing-package-data
https://docs.python.org/3/distutils/sourcedist.html#manifest
https://github.com/kennethreitz/setup.py
"""