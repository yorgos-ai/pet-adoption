#!/bin/bash

echo "This bash script automates the setup process for the pet-adoption project."

user=$(whoami)

project_dir=$(pwd)
echo "Current working directory: $project_dir"

# Check if the current directory is 'pet-adoption'
if [[ $(basename "$project_dir") = "pet-adoption" ]]; then
    # Go back one parent folder
    cd ..
    # Print the new working directory
    base_dir=$(pwd)
    echo "You are in the base working directory: $base_dir"
else
    echo "Please clone the 'pet-adoption' Git repository and run the make command from the 'pet-adoption' directory."
fi

# Install AWS CLI
echo "Installing AWS CLI..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install unzip
unzip awscliv2.zip
sudo ./aws/install
echo "AWS CLI version $(aws --version) installed successfully."

# Install Docker
echo "Installing Docker..."
sudo apt update
sudo apt install docker.io
sudo groupadd docker
sudo usermod -aG docker ${USER}
echo "Docker version $(docker --version) installed successfully."

# Install Docker Compose
echo "Installing Docker Compose..."
wget https://github.com/docker/compose/releases/download/v2.29.1/docker-compose-linux-x86_64 -o "docker-compose"
echo "Docker Compose version $(docker-compose --version) installed successfully."

# Install Pyenv
echo "Installing Pyenv..."
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev libpq-dev
curl https://pyenv.run | bash
echo -e 'export PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'eval "$(pyenv init --path)"\neval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
echo "Pyenv version $(pyenv --version) installed successfully."

# Install Poetry
echo "Installing Poetry..."
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="/home/$(user)/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
echo "Poetry version $(poetry --version) installed successfully."

# Install Python 3.10.12
echo "Installing Python 3.10.12..."
pyenv install 3.10.12
cd pet-adoption
pyenv local 3.10.12
pyenv shell

# Install project dependencies
echo "Installing project dependencies..."
poetry env use 3.10.12
poetry shell
poetry install
pre-commit install
echo "Project dependencies installed successfully."
