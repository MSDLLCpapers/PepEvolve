import setuptools

setuptools.setup(
    name="pepevolve",
    version="0.0.1",
    long_description_content_type="text/markdown",
    url="https://github.com/MSDLLCpapers/PepEVOLVE.git",
    packages=setuptools.find_packages(exclude='unit_tests'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9,<3.11',
)