# setup.py — fallback for pip < 21.3 / setuptools < 42 that can't read pyproject.toml
# On any modern pip (>=21.3) this file is ignored; pyproject.toml takes over.
from setuptools import setup, find_packages

setup(
    name="aipack",
    version="0.1.0",
    description="AI context compressor for local LLMs — Llama 3, Mistral, Phi, etc.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    packages=find_packages(include=["aipack", "aipack.*"]),
    entry_points={"console_scripts": ["aipack=aipack.cli:main"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Environment :: Console",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
