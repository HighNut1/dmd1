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
    # 1. 准备模型和设备
    print("正在加载完全体 DeepPrint_TexMinu_512 模型...")
    extractor = EXTRACTORS["DeepPrint_TexMinu_512"].load()
    model = extractor.model
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
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
        
        # 稳妥地解析出 Subject ID (比如从 'Identifier(0, 1)' 中提取出 '0')
        subject_id = str(id_obj).split(',')[0].split('(')[-1].strip()
        labels.append(subject_id)
        
        img = dataset.get(id_obj).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img)
            # 拼接 512 维特征
            feat = torch.cat([out.texture_embeddings.flatten(), out.minutia_embeddings.flatten()], dim=0)
            # 对特征进行 L2 归一化，这样后面的点积就等价于余弦相似度
            feat = F.normalize(feat, p=2, dim=0)
            embeddings.append(feat.cpu().numpy())

    # 转换为 NumPy 矩阵以进行极速运算
    X = np.array(embeddings)  # Shape: (3982, 512)
    y = np.array(labels)      # Shape: (3982,)

    # 4. 极速计算相似度矩阵
    print("\n⚡ 正在进行矩阵运算，完成近 800 万次的两两交叉比对...")
    # X 乘以 X 的转置，瞬间得到 3982 x 3982 的余弦相似度矩阵
    sim_matrix = np.dot(X, X.T)
    
    # 我们只需要矩阵的上三角部分（去掉对角线），因为 A比B 和 B比A 是一样的，且不能自己比自己
    rows, cols = np.triu_indices(X.shape[0], k=1)
    
    # 提取出所有的相似度得分
    scores = sim_matrix[rows, cols]
    
    # 生成对应的真值标签：同一个人是 1 (Genuine)，不同人是 0 (Imposter)
    true_labels = (y[rows] == y[cols]).astype(int)
    
    genuine_count = np.sum(true_labels == 1)
    imposter_count = np.sum(true_labels == 0)
    print(f"统计完毕：包含 {genuine_count} 对真匹配 (同人)，{imposter_count} 对假匹配 (不同人)。")

    # 5. 计算 ROC 曲线 和 EER
    print("\n📊 正在计算 EER (等错误率)...")
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    fnr = 1 - tpr
    
    # EER 的定义是 FNR (漏识率) 和 FPR (误识率) 极其接近的那个点
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_index]
    best_threshold = thresholds[eer_index]
    
    print("=" * 50)
    print(f"🎉 最终评测结果 (NIST14 测试集):")
    print(f"等错误率 (EER): {eer * 100:.3f}%")
    print(f"最佳区分阈值 (Threshold): {best_threshold:.4f}")
    print("=" * 50)

    # 6. 绘制并保存 ROC 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (EER = {eer*100:.2f}%)')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Fingerprint Matching ROC on NIST14 (DeepPrint 512-dim)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    save_path = "/root/autodl-tmp/fixed-length-fingerprint-extractors/roc_curve_nist14.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n🖼️ ROC 曲线图已保存至: {save_path}")
    print("你可以从 AutoDL 网页端将其下载下来放到你的汇报 PPT 里！")

if __name__ == "__main__":
    main()