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
sudo apt install docker-compose
echo "Docker Compose version $(docker-compose --version) installed successfully."

# Install Pyenv
echo "Installing Pyenv..."
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev libpq-dev
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo "Pyenv installed successfully."

# Install Poetry
echo "Installing Poetry..."
curl -sSL https://install.python-poetry.org | python3 -
line_to_add='export PATH="/home/ubuntu/.local/bin:$PATH"'
# Backup the original .bashrc file
cp ~/.bashrc ~/.bashrc.backup
# Insert the new line at the top of the .bashrc file
sed -i '1i  export PATH="\/home\/ubuntu\/.local\/bin:\$PATH"' .bashrc
echo "Poetry installed successfully."

source ~/.bashrc

echo "Setup completed successfully."
