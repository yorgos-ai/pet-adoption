## Pre-requisites

This document contains detailed instructions on how to do the initial set up that is required before you can run the code.

### 1. Create an IAM user
Create an AWS IAM user and grant administrator access to that role. Make sure to save the `AWS_SECRET_ACCESS_KEY` and `AWS_ACCESS_KEY_ID` of the IAM role you created since they will be used to configure the AWS profile for that role.

Detailed instructions can be found [here](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html).

### 2. Create an EC2 instance
Log in in the AWS concole using the IAM role you created in the previous step. Navigate to EC2 service and click on `Instances` and then `Launch an instance`.
1. Add a name for your EC2 instance.
2. Go to the Ubuntu OS images and from the drop down menu select the `Ubuntu Server 22.04` image.
3. Select `t2.xlarge` as the instance type.
4. On the `Key pair (login)` panel click on `Create a new key pair`. Enter a name for your key-pair .pem file and then click on `Create key pair`. Save the .pem file on your local machine. YOu will need it later to connect from your local machine to the remote EC2 instance using SSH.
5. On the `Configure storage` panel, edit the amount of disk storage to 30 GiB.
6. Click on `Launch instance`. The EC2 instance will be launched and started automatically.

### 3. Configure SSH
Yuu can configure SSH to be able to access the EC2 instance from your local machine through VS Code.
In your local mahcine, navigate to the .ssh folder.

1. Copy the .pem file you downloaded in the previous step to the .ssh directory of your local machine.
2. Create a file named `config`.
3. Add the following settings in the `config` file:
```
Host <the name of the EC2 instance you created>
    HostName <the Public IPv4 address of the EC2 instance>
    User ubuntu
    IdentityFile <the path to the .pem file on your local machine>
    StrictHostKeyChecking no
```
The final `config` file should look something like this:
```
Host ec2-host
    HostName 53.49.1.145
    User ubuntu
    IdentityFile c://Users/<your user name>/.ssh/example.pem
    StrictHostKeyChecking no
```

4. Open VS code in your local machine and install the `SSH remote` extension.
5. After the extension is installed, click on `Remote` and then `Connect to Host` and select the name of host you created in the config file (for example `ec2-host`).

### 4. Create S3 buckets
This project uses two S3 buckets. The `pet-adoption-mlops` is a general purpose S3 bucket for storing artifacts and the second, `mlflow-artifacts-pet-adoption` is the artifact folder for the MLflow tracking server.

In the AWS console, navigate to the `S3` service and create the following two S3 buckets:
- pet-adoption-mlops
- mlflow-artifacts-pet-adoption

> [!NOTE]
> The name of the S3 buckets should be unique within an AWS region. You can add a suffix like `-test` or something similar on the S3 bucket names to make them unique.

### 5. Clone the project repository using HTTPS
Go to VS code which is connected with SSH to the EC2 instance, <ins>open a new terminal</ins> and copy paste the following command:
```
git clone https://github.com/yorgos-ai/pet-adoption.git && \
cd pet-adoption/ && code .
```
A new VS code window will pop up and the repository files and folders should be visible in the left panel.

### 6. Create a `.env` file
This project uses a .env file to store credentials and other environment variables. These environment variables are loaded at run time using the `dotenv` Python package.

Rename the `.env.example` file to `.env` and change the name of the S3 buckets to the names you used in step 4.

### 7. Run the initial setup script.
Install `make` in your EC2 instance to be able to execute make commands.
```
sudo apt update && sudo apt install make
```

Once the installation is finished, run the set up script.
```
make init_setup
```

> [!NOTE]
> During the execution of the set_up.sh script, you might be prompted to restart some services. Press Enter a few times and the script will continue.

This make command executes the [set_up.sh](pet_adoption/scripts/set_up.sh) script which does the following things:
- Installs the AWS CLI
- Installs Docker
- Installs Docker Compose
- Installs and configures Pyenv
- Installs and configures Poetry

### 8. Install Python dependencies
Open a new terminal and execute the following command:
```
make env_setup
```
This make command installs Python 3.0.12 using Pyenv, creates a Poetry environment and installs all the necessary Python packages used in this project.

### 9. Configure AWS credentials
Set up your AWS profile by adding your `AWS_SECRET_ACCESS_KEY` and `AWS_ACCESS_KEY_ID` form step 1 to your AWS profile. These credentials will be automatically used to authenticate your IAM role and be able access AWS services from your EC2 instance.

```
aws configure --profile mlops-zoomcamp
```
Copy-paste the secret access key and the access key id to the terminal. Also, add your AWS region as the default region in your AWS profile and set the default output format to json. Your AWS profile should look something like this:
```
AWS Access Key ID: XXXXXXXXX
AWS Secret Access Key: XXXXXXXXX
Default region name: eu-west-1
Default output format: json
```
