from setuptools import setup, find_packages

setup(
    name="png_metadata_tools",
    version="0.1.0",
    author="vosreth",
    author_email="vosreth@gmail.com",
    description="A sophisticated system for PNG metadata operations with British standards",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vosreth/png-metadata-tools",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pillow>=11.1.0",
        "numpy>=2.2.3",
        "psutil>=7.0.0",
    ],
    extras_require={
        "viewer": ["tkinterdnd2>=0.4.2"],
        "dev": [
            "pytest>=8.3.4",
            "pytest-cov>=6.0.0",
            "coverage>=7.6.12",
            "memory-profiler>=0.61.0",
        ],
    },
)