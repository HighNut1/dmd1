import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from flx.setup.experiments import EXTRACTORS
from flx.setup.datasets import get_nist14

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 准备模型：先加载框架基础，然后强行覆盖为我们微调后的权重
    print("正在初始化网络结构...")
    extractor = EXTRACTORS["DeepPrint_TexMinu_512"].load()
    model = extractor.model
    
    finetuned_model_path = "/root/autodl-tmp/fixed-length-fingerprint-extractors/models/DeepPrint_TexMinu_512/finetuned_nist14_model.pyt"
    print(f"正在加载专属微调权重: {finetuned_model_path}...")
    checkpoint = torch.load(finetuned_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.to(device)
    model.eval()
    
    # 2. 加载数据集
    print("正在初始化 NIST14 数据集...")
    nist14_path = "/root/autodl-tmp/fixed-length-fingerprint-extractors/data/fingerprints/NIST14"
    dataset = get_nist14(nist14_path)
    total_samples = len(dataset.ids)
    
    embeddings = []
    labels = []

    # 3. 提取全量特征
    print(f"\n🚀 开始提取全部 {total_samples} 张指纹的特征 (可能需要几分钟)...")
    for i in tqdm(range(total_samples), desc="特征提取进度"):
        id_obj = dataset.ids[i]
        subject_id = str(id_obj).split(',')[0].split('(')[-1].strip()
        labels.append(subject_id)
        
        img = dataset.get(id_obj).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img)
            feat = torch.cat([out.texture_embeddings.flatten(), out.minutia_embeddings.flatten()], dim=0)
            feat = F.normalize(feat, p=2, dim=0)
            embeddings.append(feat.cpu().numpy())

    X = np.array(embeddings)  # Shape: (3982, 512)
    y = np.array(labels)      # Shape: (3982,)

    # 4. 极速计算相似度矩阵
    print("\n⚡ 正在进行矩阵运算，完成近 800 万次的两两交叉比对...")
    sim_matrix = np.dot(X, X.T)
    rows, cols = np.triu_indices(X.shape[0], k=1)
    scores = sim_matrix[rows, cols]
    true_labels = (y[rows] == y[cols]).astype(int)
    
    # 5. 计算 ROC 曲线 和 EER
    print("\n📊 正在计算 EER (等错误率)...")
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    fnr = 1 - tpr
    
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_index]
    best_threshold = thresholds[eer_index]
    
    print("=" * 50)
    print(f"🎉 最终评测结果 (微调后的 NIST14):")
    print(f"等错误率 (EER): {eer * 100:.3f}%")
    print(f"最佳区分阈值 (Threshold): {best_threshold:.4f}")
    print("=" * 50)

    # 6. 绘制并保存新的 ROC 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='red', lw=2, label=f'Finetuned ROC (EER = {eer*100:.2f}%)')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Fingerprint Matching ROC on NIST14 (Finetuned Model)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    save_path = "/root/autodl-tmp/fixed-length-fingerprint-extractors/roc_curve_nist14_finetuned.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n🖼️ 新的 ROC 曲线图已保存至: {save_path}")

if __name__ == "__main__":
    main()