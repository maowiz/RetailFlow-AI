# Contributing to RetailFlow AI

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)
- Error messages/logs

### Suggesting Features

Feature requests are welcome! Please include:
- Clear description of the feature
- Use case / business value
- Example implementation (if applicable)

### Pull Requests

1. **Fork the repository**
   ```bash
   git fork https://github.com/maowiz/RetailFlow-AI.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow existing code style
   - Add tests if applicable
   - Update documentation

4. **Commit your changes**
   ```bash
   git commit -m "feat: add amazing feature"
   ```
   
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `refactor:` Code refactoring
   - `test:` Adding tests
   - `chore:` Maintenance tasks

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Describe your changes
   - Link related issues
   - Add screenshots (for UI changes)

## 📝 Code Style

- **Python**: Follow PEP 8
- **Docstrings**: Use Google-style docstrings
- **Type hints**: Add type annotations for functions
- **Comments**: Explain "why", not "what"

Example:
```python
def calculate_safety_stock(
    demand_std: float,
    lead_time: int,
    service_level: float = 0.95
) -> float:
    """Calculate safety stock using statistical formula.
    
    Args:
        demand_std: Standard deviation of demand
        lead_time: Supplier lead time in days
        service_level: Target service level (0-1)
        
    Returns:
        Safety stock quantity
    """
    z_score = stats.norm.ppf(service_level)
    return z_score * demand_std * np.sqrt(lead_time)
```

## 🧪 Testing

Before submitting a PR:

```bash
# Run all tests
pytest tests/

# Run API tests
python test_api.py

# Check code style
flake8 src/

# Format code
black src/
```

## 📚 Documentation

When adding features:
- Update README.md if needed
- Add docstrings to new functions/classes
- Update API docs for new endpoints
- Add usage examples

## 🎯 Good First Issues

Look for issues labeled `good-first-issue` — these are great starting points!

## 💡 Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/RetailFlow-AI.git
cd RetailFlow-AI

# Add upstream remote
git remote add upstream https://github.com/maowiz/RetailFlow-AI.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dev dependencies
pip install -r requirements-dev.txt
```

## 🚀 Areas for Contribution

**High Priority:**
- Unit tests for models & optimization
- Performance benchmarks
- Cloud deployment guides (AWS, GCP, Azure)
- Mobile-responsive dashboard

**Medium Priority:**
- Additional ML models (LSTM, Transformers)
- Database integration (PostgreSQL, MongoDB)
- User authentication
- Export to Excel/PDF

**Nice to Have:**
- Multilingual support
- Dark/light theme toggle
- A/B testing framework
- Automated reporting

## ❓ Questions?

- Open a [Discussion](https://github.com/maowiz/RetailFlow-AI/discussions)
- Contact: maowiz@example.com

## 📜 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Assume good intentions

Thank you for contributing! 🎉
