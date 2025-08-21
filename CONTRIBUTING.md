# Contributing to OpenQBench

Thanks for your interest in contributing to OpenQBench! We welcome contributions from everyone and appreciate your help in making this project better.

This document outlines how you can contribute, from reporting bugs to submitting new features. Please take a moment to review it before getting started.

## Setting up
1. Fork the repository.
2. Clone your fork.
```bash
git clone https://github.com/yourusername/open-qbench.git
cd open-qbench
```
3. Add the original repository as an upstream remote to pull updates.
```bash
git remote add upstream https://github.com/PCSS-Quantum/open-qbench.git
```
4. Create a virtual environment and install dependencies.
```bash
uv venv
uv sync --dev
```
5. Install `pre-commit` hooks. We use pre-commit to ensure code quality and consistent formatting.
```bash
pre-commit install
```
This will run checks like linting and formatting automatically before each commit.

6. Verify setup by running tests.
```bash
uv run pytest tests/test_benchmarks.py
```

## Contributing code
1. Create a new branch for your work: `git checkout -b feature/your-feature-name`.
2. Write clean, well-documented code.
    - Your code will be automatically formatted and checked by `pre-commit` hooks. If any checks fail, it will prevent the commit and often fix issues automatically. Just `git add` the fixed files and try to commit again.
3. Include unit tests for new features and bug fixes.
4. Ensure all existing tests pass.
> [!NOTE]
> If you don't have access to `ptseries` you can skip tests that use it.
5. Push your branch.
6. Open a Pull Request (PR) from your GitHub fork.


## Style and lint
We use Ruff for linting and code formatting. Use of MyPy is recommended for static type checking to ensure type consistency throughout the project. For the configuration of these tools, look into the `pyproject.toml` file.

If you are unsure about the style or structure of your code, don't worry too much. Maintainers will help you organize your contribution.
