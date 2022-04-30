FROM nvidia/cuda:11.4.2-devel-ubuntu20.04
# FROM ubuntu:20.04

# コンテナにプロキシ環境変数を追加する(学内)
# ENV https_proxy "http://wwwproxy.kanazawa-it.ac.jp:8080"
# ENV http_proxy "http://wwwproxy.kanazawa-it.ac.jp:8080"

# ROSのインストール時にtzdataのタイムゾーン選択で止まらないように
ENV DEBIAN_FRONTEND=noninteractive

# aptを日本のミラーに設定
# 参考: http://www.aise.ics.saitama-u.ac.jp/~gotoh/HowToInstallUbuntu1804OnWSL.html
RUN sed -i.bak -e "s/http:\/\/archive\.ubuntu\.com/http:\/\/jp\.archive\.ubuntu\.com/g" /etc/apt/sources.list


RUN apt-get update

# sudo:sudo
# x11-apps:xeyes用
# net-tools:ifconfig用
# mesa-utils:glxgears用
# gnupg:rosのadd-key用
RUN apt-get install -y curl \
    sudo \
    gnupg \
    x11-apps \
    iputils-ping \
    net-tools \
    mesa-utils \
    vim \
    dbus-x11

# Pytorch等々のインストール
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y libjpeg-dev zlib1g-dev
RUN pip3 install torch
RUN pip3 install torchvision
RUN pip3 install matplotlib

# aptのキャッシュを削除してイメージの容量削減
RUN rm -rf /var/lib/apt/lists/
