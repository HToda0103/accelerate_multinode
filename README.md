# accelerateでマルチノード学習
- accelerate_config
  - accelerateのコンフィグ
- docker-compose_main.yaml
  - mainのマシンで実行
- docker-compose_sub.yaml
  - subのマシンで実行   
- [Zenn_url](https://zenn.dev/hiyoo/articles/e312dab357421b)
# 実行環境
| machine | ubuntu | GPU1 | GPU2 | Cuda | IP |
| ---- | ---- | ---- | ---- | ---- | ---- |
| main | 24.04 | TITAN X (Pascal) 12G | TITAN X (Pascal) 12G| cuda:12.6(nvidia-smi),<br>DriverVersion:560.35.03 | 10.0.0.3 |
| sub | 24.04 | Quadro RTX 8000 48G (7.5) | Quadro RTX 8000 48G (7.5) | cuda:12.6(nvidia-smi),<br>DriverVersion:560.35.03 | 10.0.0.6 |
# 環境構築  
## 鍵作成　　
- 鍵をmainマシンで作成
   ```bash
   ssh-keygen -t rsa -f ./id_rsa
   ```
## dockerコンテナの作成
- main,subマシンでイメージをbuildする
   ```bash
   docker build -t accelerate_img .
   ```
- main,subマシンので実行
  - mainのdockerコンテナ作成
    ```bash
    docker compose -f docker-compose_main.yaml up -d
    ```
  - subのdockerコンテナ作成
    ```bash
    docker compose -f docker-compose_sub.yaml up -d
    ```
# プログラム実行
## 1node2GPU
```bash
accelerate launch classifier_acceraleter_BERT.py
```
## 2node4GPU
- それぞれのマシンで使用している`ネットワークインタフェース`を指定
- mainマシンで実行
  ```bash
  NCCL_SOCKET_IFNAME=eno1 \
  accelerate \
  launch \
  classifier_acceraleter_BERT.py
  ```
- subマシンで実行
  ```bash
  NCCL_SOCKET_IFNAME=enp4s0 \
  accelerate \
  launch \
  classifier_acceraleter_BERT.py
  ```
