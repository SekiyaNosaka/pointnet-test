# **PointNet-Test**

## 概要
PointNetを見よう見まねで自作したもの。

原著ではセグメンテーションタスクとして紹介しているが、ここでは固定サイズベクトルの回帰モデルとして実装した。

全く不完全なので参考程度なり、拡張するなりのヒントになれば...

**手法詳細**
- PointNet(CVPR 2017)にインスパイアされた点群深層学習モデル
  - 入力: クロップ点群
  - 出力: 3つの情報をもつベクトル (Roll, Pitch, Yawを模倣)


---
## 作成者
野坂 碩也


---
## 動作確認済み実行環境
- OS: Ubuntu 18.04, 20.04
- Python2系
    - numpy
    - pytorch==1.4
    - matplotlib
- Python3系
    - numpy
    - pytorch==1.11
    - matplotlib

---
## ビルド・実行方法
1. まずDocker環境をビルド及びラン
```
# イメージビルド
./build-docker-image.sh

# コンテナ起動
./run-docker-container.sh

# コンテナ停止
./stop-docker-container.sh
```

2. テスト動作
```
python3 check.py
```

3. 3次元ベクトルがnumpy形式で表示されればOK


---
## ネットワークの学習

### Parametor
| パラメータ | 内容 |
| :---- | ----: |
| EPOCH | 学習エポック数 |
| BATCH_SIZE | 学習バッチサイズ <br> (推奨 2^n) |
| NUM_POINTS | 入力点群数(default: 512) |
| NUM_LABELS | 出力ベクトルサイズ(default: 3) |


---
#### TODO
- 現ネットワークでは、点群の不変性を考慮したアフィン行列機構を実装しきれていない
- Activation(tanhなど)を最後に噛ませた時と比較
- torch部分のWarning表示がされないコードに改変
  - `/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:7: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).`
  - つまり，テンソル変数をclone・detachする操作をすれば良い
- 現状だと、総Dataset数とBatchSizeの剰余が1になる時に，学習内でのBatchNorm部分でエラーを起こしてしまう
- 最適化関数の学習率やモデルの層数，次元数等のハイパーパラメータ調整の検討(時間に余裕があれば)
- optunaを用いたハイパーパラメータを調整の検討(PointNetベースにしていて秀逸なのでやる必要もないか)

