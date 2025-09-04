#!/bin/bash
# Activate the Python virtual environment

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
eval "$(pyenv virtualenv-init -)"

# Activate the virtual environment
pyenv activate rag-env

echo "Python virtual environment 'rag-env' activated!"
echo "Python version: $(python --version)"
echo "Pip packages installed: $(pip list | wc -l) packages"
