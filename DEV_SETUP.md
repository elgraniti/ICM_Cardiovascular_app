# Development Setup

For developers who want to contribute or modify the code:

## Setup

```bash
# 1. Clone repository
git clone https://github.com/yourusername/ICM_Cardiovascular_app.git
cd ICM_Cardiovascular_app

# 2. Create development environment
conda create -n icm-cardio-dev python=3.9
conda activate icm-cardio-dev

# 3. Install in development mode
pip install -e .[dev]
```

## Development Features

This gives you access to:

### Both Usage Methods
- **Direct scripts**: `python database_creator.py`
- **CLI tools**: `icm-database-creator`

### Development Tools
- **pytest**: Run tests with `pytest`
- **black**: Code formatting with `black .`
- **flake8**: Linting with `flake8`
- **mypy**: Type checking with `mypy icm_cardiovascular/`

### Hot Reloading
- Changes to the code are immediately reflected
- No need to reinstall after modifications

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=icm_cardiovascular

# Run specific test file
pytest test_install.py
```

## Code Quality

```bash
# Format code
black .

# Check linting
flake8

# Type checking
mypy icm_cardiovascular/
```
