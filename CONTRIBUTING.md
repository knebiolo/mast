# Contributing to MAST

Thank you for your interest in contributing to MAST (Movement Analysis Software for Telemetry)!

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a branch** for your changes: `git checkout -b feature/your-feature-name`
4. **Make your changes**
5. **Test your changes** thoroughly
6. **Commit your changes**: `git commit -m "Description of changes"`
7. **Push to your fork**: `git push origin feature/your-feature-name`
8. **Submit a Pull Request** on GitHub

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Installation for Development

```bash
# Clone the repository
git clone https://github.com/knebiolo/mast.git
cd mast

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install in development mode with dependencies
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8
```

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use meaningful variable names (snake_case for variables/functions)
- Add docstrings to all public functions and classes
- Keep line length to 100 characters or less
- Use type hints where appropriate

## Testing

Before submitting a pull request:

1. **Run existing tests**: `pytest tests/`
2. **Add tests** for new functionality
3. **Ensure code coverage** doesn't decrease

## Documentation

- Update the README.md if you change functionality
- Add docstrings to new functions and classes
- Update CHANGELOG.md with your changes
- Include examples for new features

## Reporting Issues

When reporting bugs, please include:

- MAST version
- Python version
- Operating system
- Complete error message/traceback
- Minimal code example to reproduce the issue
- Expected vs actual behavior

## Suggesting Features

Feature requests are welcome! Please:

- Check if the feature has already been requested
- Describe the use case clearly
- Explain why this feature would be useful to other users
- Provide examples if possible

## Code Review Process

1. Maintainers will review your PR within a week
2. Address any requested changes
3. Once approved, your PR will be merged
4. Your contribution will be credited in the changelog

## Questions?

- Open an issue on GitHub
- Email: kevin.nebiolo@kleinschmidtgroup.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for making MAST better! 🐟📡
