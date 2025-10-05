# ベースとなるPyTorchやCUDAのイメージを指定
# FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# SSHサーバーとsudoをインストール
RUN apt-get update && apt-get install -y openssh-server sudo
# 必要なPython3とpipをインストール
RUN apt-get update && apt-get install -y python3 python3-pip
# SSH接続を許可するための設定
RUN mkdir /var/run/sshd
# rootユーザーでのSSHログインを許可
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSHポートを開放
EXPOSE 22

# 必要なPythonライブラリをインストール
RUN pip install transformers accelerate pandas numpy

# コンテナ起動時にSSHサーバーを開始する
CMD ["/usr/sbin/sshd", "-D"]