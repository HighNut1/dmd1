import torch
import torch.nn.functional as F
from flx.setup.experiments import EXTRACTORS
from flx.setup.datasets import get_nist14

def match_fingerprints(model, device, dataset, id_A, id_B, label):
    print(f"\n[{label}] 开始比对:")
    print(f"图 A ID: {id_A}")
    print(f"图 B ID: {id_B}")
    
    # 【核心修复】使用框架内置的 dataset.get() 方法
    # 它会自动完成 ToTensor(), 像素值归一化 (0~1) 等所有官方要求的预处理
    img1 = dataset.get(id_A).unsqueeze(0).to(device)
    img2 = dataset.get(id_B).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out1 = model(img1)
        out2 = model(img2)
        
        # 提取 512 维特征
        feat1 = torch.cat([out1.texture_embeddings.flatten(), out1.minutia_embeddings.flatten()], dim=0)
        feat2 = torch.cat([out2.texture_embeddings.flatten(), out2.minutia_embeddings.flatten()], dim=0)
        
    # 计算真正的余弦相似度
    cos_sim = F.cosine_similarity(feat1, feat2, dim=0).item()
    display_score = max(0, cos_sim) * 100
    
    print("-" * 40)
    print(f"余弦相似度: {cos_sim:.4f}")
    print(f"相似度得分: {display_score:.2f}%")
    
    # 浮点数余弦相似度的阈值通常根据 ROC 曲线决定，通常在 0.35 ~ 0.5 之间
    if cos_sim > 0.40:
        print("💡 系统判定: 【匹配成功】 是同一个人！")
    else:
        print("🚫 系统判定: 【匹配失败】 不是同一个人。")
    print("-" * 40)

def main():
    print("正在加载官方预训练 DeepPrint_TexMinu_512 模型...")
    extractor = EXTRACTORS["DeepPrint_TexMinu_512"].load()
    model = extractor.model
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("模型准备就绪。\n" + "="*50)

    print("正在加载 NIST14 数据集流水线 (应用官方正确的图像预处理)...")
    nist14_path = "/root/autodl-tmp/fixed-length-fingerprint-extractors/data/fingerprints/NIST14"
    dataset = get_nist14(nist14_path)

    try:
        # 自动在数据集中搜索名字里带 '0001' 和 '0002' 的样本标识符
        person1_ids = [i for i in dataset.ids if "0001" in str(i)]
        person2_ids = [i for i in dataset.ids if "0002" in str(i)]
        
        # 提取同一个人 (0001) 的两张图，以及不同人 (0002) 的一张图
        id_f_1 = person1_ids[0]
        id_s_1 = person1_ids[1] if len(person1_ids) > 1 else person1_ids[0]
        id_f_2 = person2_ids[0]
        
        match_fingerprints(model, device, dataset, id_f_1, id_s_1, "真匹配 (同一个人)")
        match_fingerprints(model, device, dataset, id_f_1, id_f_2, "假匹配 (不同人)")
        
    except Exception as e:
        print(f"\n自动匹配ID失败 ({e})，使用数据集前三张图进行兜底测试...")
        match_fingerprints(model, device, dataset, dataset.ids[0], dataset.ids[1], "测试一 (图1 vs 图2)")
        match_fingerprints(model, device, dataset, dataset.ids[0], dataset.ids[2], "测试二 (图1 vs 图3)")

if __name__ == "__main__":
    main()