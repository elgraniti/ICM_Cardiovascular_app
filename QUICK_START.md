# Quick Start Guide

## Two Ways to Use ICM Cardiovascular App

### 🚀 Method 1: Simple Direct Scripts (Recommended for Conda users)

```bash
# 1. Set up conda environment
conda create -n icm-cardio python=3.9
conda activate icm-cardio

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the apps
python database_creator.py    # Creates databases at localhost:8050
python icm_database.py        # Views databases at localhost:8055
```

### 📦 Method 2: Pip Package Installation

```bash
# 1. Install package
pip install .

# 2. Use command-line tools
icm-database-creator    # Creates databases at localhost:8050
icm-database-viewer     # Views databases at localhost:8055
```

## That's it! 

Both methods give you the same functionality:
- **Database Creator**: Web app for creating databases from ICM+ files
- **Database Viewer**: Web app for exploring and analyzing the databases

Choose the method that fits your workflow best!
