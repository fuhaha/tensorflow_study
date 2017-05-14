# Ubuntu16.04에서 GPU 기반의 tensorflow 설치하기
작성자 : 전성욱
작성일 : 2017/03/28
email : codetree@gmail.com

---

대상장비
CPU : Xeon CPU
RAM : 32G
HDD : SSD 250
GPU : Nvidia 750ti
OS : Ubuntu16.04 Server

## 0.OS 기본 환경 설정
### How To Add a User : 사용자 추가
~~~
sudo adduser student
~~~
### Add the New User to the Sudo Group
~~~
sudo usermod -aG sudo student
~~~
### How do I install Python 3.6 (Ubuntu 14.04 and 16.04)
tensorflow site에서는 3.5기준이므로 선택사항이다.
~~~
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update

sudo apt-get install python3.6
cd /usr/bin/
sudo ln -s python3.6 python3
sudo ln -s python3.6m python3m

sudo apt-get install python3-pip
~~~
## 1.Install GPU Driver 

### 1) 설치된 VGA확인
~~~
lshw -numeric -C display
~~~
#### or
~~~
lspci -vnn | grep VGA
~~~
### 2) NVIDIA Site에서 Driver확인...
http://www.nvidia.com/Download/index.aspx

Version:	375.39
Release Date:	2017.2.14
Operating System:	Linux 64-bit
Language:	English (US)
File Size:	73.68 MB

### 3) INSTALL NVIDIA Driver
~~~
sudo apt-get install nvidia-375
~~~

## 2. CUDA Toolkit 설치
Download Installer for Linux Ubuntu 16.04 x86_64
~~~
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
~~~
The base installer is available for download below.
Base Installer	Download (2.6 KB)  
Installation Instructions:
~~~
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
~~~

Edit .bashrc
~~~
vi .bashrc
~~~
내용 추가
~~~
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${CUDA_HOME}/lib64"
export PATH=${PATH}:/usr/local/cuda/bin
~~~

## 3. Install CUDNN
1. https://developer.nvidia.com/cudnn 접속. Download 버튼을 통해 설치 (NVIDIA 가입 필요)
2. 다운로드 받은 파일을 우클릭하여 압축을 품. CUDA 폴더가 생성됨.
3. 다음을 입력하여 관리자 권한으로 탐색기 실행. usr/local/cuda 로 진입

cuDNN Download
Download cuDNN v5.1 (Jan 20, 2017), for CUDA 8.0
cuDNN v5.1 Library for Linux
	URL : https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod_20161129/8.0/cudnn-8.0-linux-x64-v5.1-tgz

~~~
cd /usr/local
sudo tar xvzf ~/downloads/cudnn-8.0-linux-x64-v5.1.tgz 
# OR
tar xvzf cudnn-8.0-linux-x64-v5.1.tgz
sudo cp -Prv ./cuda/lib64/* /usr/local/cuda/lib64/
sudo cp -Prv ./cuda/include/* /usr/local/cuda/include
~~~
~~~
sudo apt install nautilus

sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
~~~

Install NVIDIA CUDA Profile Tools Interface library
The libcupti-dev library, which is the NVIDIA CUDA Profile Tools Interface. 
This library provides advanced profiling support. 
To install this library, issue the following command:
~~~
sudo apt-get install libcupti-dev
~~~
## 4.Python3 가상환경 설정
~~~
sudo apt-get install python3-virtualenv
~~~
~~~
cd tensorflow/
virtualenv -p python3 venv
source ./venv/bin/activate
~~~

## 5.Install Tensorflow GPU
(tensorflow)$ pip install --upgrade tensorflow      # for Python 2.7
(tensorflow)$ pip3 install --upgrade tensorflow     # for Python 3.n
(tensorflow)$ pip install --upgrade tensorflow-gpu  # for Python 2.7 and GPU
(tensorflow)$ pip3 install --upgrade tensorflow-gpu # for Python 3.n and GPU



### Validate your installation
Test code
~~~
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
~~~

~~~
student@school:~$ cd tensorflow/
student@school:~/tensorflow$ source ./venv/bin/activate
(venv) student@school:~/tensorflow$ python
Python 3.6.0+ (default, Feb  4 2017, 11:11:46) 
[GCC 5.4.1 20161202] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcublas.so.8.0 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcudnn.so.5 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcufft.so.8.0 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcurand.so.8.0 locally
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:910] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties: 
name: GeForce GTX 750
major: 5 minor: 0 memoryClockRate (GHz) 1.2805
pciBusID 0000:01:00.0
Total memory: 978.50MiB
Free memory: 929.62MiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 750, pci bus id: 0000:01:00.0)
>>> print(sess.run(hello))
b'Hello, TensorFlow!'
>>> 
~~~


## 6. Installing with Docker
### 1. Install Docker on your machine as described in the Docker documentation.
OS requirements :Yakkety 16.10, Xenial 16.04 (LTS), Trusty 14.04 (LTS)

#### Uninstall old versions
~~~
sudo apt-get remove docker docker-engine
~~~

#### SET UP THE REPOSITORY
##### Docker CE

Install packages to allow apt to use a repository over HTTPS:
~~~
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
~~~

Add Docker’s official GPG key:
~~~
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
~~~

Verify that the key fingerprint is 9DC8 5822 9FC7 DD38 854A E2D8 8D81 803C 0EBF CD88.
~~~
sudo apt-key fingerprint 0EBFCD88
~~~

Use the following command to set up the stable repository.
~~~
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
~~~
#### INSTALL DOCKER

##### 1) Update the apt package index.
~~~
sudo apt-get update
~~~

##### 2) Install the latest version of Docker, or go to the next step to install a specific version. Any existing installation of Docker is replaced.
Use this command to install the latest version of Docker:
Docker Edition	Command
###### Docker CE
~~~
sudo apt-get install docker-ce
~~~
###### Docker EE
~~~
sudo apt-get install docker-ee
~~~
##### 3) On production systems, you should install a specific version of Docker instead of always using the latest. This output is truncated. List the available versions. For Docker EE customers, use docker-ee where you see docker-ce.
~~~
apt-cache madison docker-ce
~~~
Verify that Docker CE or Docker EE is installed correctly by running the hello-world image.
~~~
sudo docker run hello-world
~~~

### 2. Optionally, create a Linux group called docker to allow launching containers without sudo as described in the Docker documentation. (If you don't do this step, you'll have to use sudo each time you invoke Docker.)
Add the New User to the Sudo Group
~~~
sudo usermod -aG docker student
~~~

### 3. To install a version of TensorFlow that supports GPUs, you must first install nvidia-docker, which is stored in github.
https://github.com/NVIDIA/nvidia-docker

Installing nvidia-docker
Ubuntu distributions
Install nvidia-docker and nvidia-docker-plugin
~~~
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
~~~

Test nvidia-smi
~~~
nvidia-docker run --rm nvidia/cuda nvidia-smi
~~~

### 4. Launch a Docker container that contains one of the TensorFlow binary images.
https://hub.docker.com/r/tensorflow/tensorflow/tags/

#### CPU-only
docker run -it -p hostPort:containerPort TensorFlowCPUImage
docker run -it gcr.io/tensorflow/tensorflow bash
docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow 

docker run -it gcr.io/tensorflow/tensorflow:latest-py3 bash
docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow:latest-py3

#### GPU support
nvidia-docker run -it -p hostPort:containerPort TensorFlowGPUImage
nvidia-docker run -it gcr.io/tensorflow/tensorflow:latest-gpu bash
nvidia-docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow:latest-gpu





