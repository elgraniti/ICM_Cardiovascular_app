#!/usr/bin/env python3
"""
Setup script for ICM Plus Database Creator and Viewer Application Package
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "ICM Cardiovascular Database Creator and Viewer"

# Read requirements from requirements.txt
def read_requirements():
    requirements = []
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        # Fallback to basic requirements if file doesn't exist
        requirements = [
            "dash>=2.14.0",
            "dash-bootstrap-components>=1.5.0",
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "duckdb>=0.8.0",
            "plotly>=5.17.0",
        ]
    return requirements

setup(
    name="icm-cardiovascular",
    version="1.0.0",
    author="Victor Torres",
    author_email="victor.torreslopez@yale.edu",
    description="ICM+ File Database Creator and Viewer",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/elgraniti/ICM_Cardiovascular_app",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Database :: Database Engines/Servers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
    },
    entry_points={
        "console_scripts": [
            "icm-database-creator=icm_cardiovascular.database_creator:main",
            "icm-database-viewer=icm_cardiovascular.icm_database:main",
        ],
    },
    include_package_data=True,
    package_data={
        "icm_cardiovascular": ["*.txt", "*.md"],
    },
    keywords="icmPlus, database, medical, icm, dash, data-analysis",
    project_urls={
        "Bug Reports": "https://github.com/elgraniti/ICM_Cardiovascular_app/issues",
        "Source": "https://github.com/elgraniti/ICM_Cardiovascular_app",
    },
)
