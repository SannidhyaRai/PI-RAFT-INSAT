# Contributing Guidelines

Welcome to the PI-RAFT project! To maintain code stability and ensure high reproducibility, we follow a strict development workflow.

## Development Workflow

### 1. No Direct Main Commits
Direct modifications to the `main` or `master` branches are blocked. All changes must go through a branch-and-PR (Pull Request) workflow.

### 2. Feature Branching
Always create a new feature branch for your edits:
```bash
# Pull latest updates from main
git checkout main
git pull origin main

# Create a local development branch
git checkout -b feature-name
```

### 3. Commits and Messages
Commit your changes with clear, meaningful descriptions of what has changed:
```bash
# Add modified files
git add .

# Commit with a structured message
git commit -m "feat: implement INSAT-3DS HDF5 channel parsing stubs"
```

### 4. Push and Create a Pull Request
Push your local branch to the remote repository and create a Pull Request:
```bash
git push origin feature-name
```
*   Ensure that all tests pass locally (`pytest tests/`) before submitting the PR.
*   Ask a team member to review the code changes before merging.
