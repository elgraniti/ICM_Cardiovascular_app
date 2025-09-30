"""
ICM Cardiovascular Application Package

A comprehensive tool for creating and managing ICM+ file and location to create cardiovascular databases.
Includes both database creation and visualization capabilities.
"""

__version__ = "1.0.0"
__author__ = "ICM Cardiovascular Lab"
__description__ = "ICM+ Cardiovascular Database Creator and Viewer"

from .database_creator import main as run_database_creator
from .icm_database import main as run_database_viewer

__all__ = ['run_database_creator', 'run_database_viewer']
