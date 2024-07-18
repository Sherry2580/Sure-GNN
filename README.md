# GNN-based Fair Classification without Sensitive Attributes
This project compares the impact of fairness analysis on model training using two different approaches **SURE_GNN** and **SURE_NN**. Both approaches implement the SURE (Significant Unfairness
Risk Elimination) fairness analysis algorithm but apply it to different types of neural network models.

## Run the code
**SURE_GNN :**

- Summary : **SURE_GNN** 結合了圖卷積神經網絡(GCN)和公平性分析算法(SURE)，處理圖結構數據並進行公平性評估。

- Prepare Dataset：確保 `data_rem3.py` 已經成功執行並生成了 `cora_data2.npz` 文件。

- Train & Test：執行以下命令來運行
```bash
python test.py --dataset cora_data2 --dropout_p 0.5 --epochs 200 --num_procs 5 --num_seeds 5
```
**SURE_NN :**

- Summary : **SURE_NN** 使用多層前饋神經網絡（MLP）來處理由圖數據結構轉成表格式數據的 Cora 資料集，同時結合公平性分析算法(SURE)進行公平性評估。

- Prepare Dataset：確保 `data_rem.py` 已經成功執行並生成了 `cora_test.npz` 文件。

- Train & Test：執行以下命令來運行
```bash
python test.py --dataset cora_test --dropout_p 0.5 --epochs 200 --num_procs 5 --num_seeds 5
```

### Reference:
<pre>
SURE: Robust, Explainable, and Fair Classification without Sensitive Attributes,
by D. Chakrabarti,
in KDD 2023.
</pre>
