import torch
from tqdm import tqdm
from flx.setup.experiments import EXTRACTORS
from flx.setup.datasets import get_nist14

def main():
    # 1. 加载模型并放到 GPU 上（为了提速）
    print("正在加载训练好的 DeepPrint_Tex_512 模型...")
    extractor = EXTRACTORS["DeepPrint_Tex_512"].load()
    model = extractor.model
    model.eval()
    
    # 自动检测 GPU，如果可用就用 GPU 跑，会快很多
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"模型已加载至: {device}")

    # 2. 读取 NIST14 真实数据集
    print("正在加载 NIST14 真实数据集...")
    nist14_path = "/root/autodl-tmp/fixed-length-fingerprint-extractors/data/fingerprints/NIST14"
    dataset = get_nist14(nist14_path)

    total_samples = len(dataset.ids)
    is_binary_count = 0
    non_binary_count = 0

    print(f"\n开始遍历并检测全部 {total_samples} 张图片的特征向量...")
    print("严格标准: 向量中的 512 个维度必须全部等于 1.0 或 -1.0")
    
    # 3. 逐个提取并检测
    for i in tqdm(range(total_samples), desc="检测进度"):
        identifier = dataset.ids[i]
        
        # 读取单张图片并增加 Batch 维度，送入对应的计算设备
        img = dataset.get(identifier)
        input_tensor = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            
        features = output.texture_embeddings
        
        # 4. 核心检测逻辑：检查张量里所有的值是否只由 1.0 和 -1.0 组成
        # 使用 torch.all 确保 512 个维度全满足条件
        is_strictly_binary = torch.all((features == 1.0) | (features == -1.0)).item()
        
        if is_strictly_binary:
            is_binary_count += 1
        else:
            non_binary_count += 1
            # 如果发现第一个非二值化的向量，打印出来看看是怎么回事
            if non_binary_count == 1:
                print(f"\n[警告] 发现非二值化向量！(来自ID: {identifier})")
                print(f"它的前10个值为: {features[0][:10].cpu().numpy()}")

    # 5. 打印最终统计结果
    print("\n" + "="*50)
    print("检测完成！最终统计报告：")
    print(f"总计检测图片: {is_binary_count + non_binary_count} 张")
    print(f"✅ 完全二值化 (仅含 1.0 和 -1.0) 的数量: {is_binary_count}")
    print(f"❌ 非二值化 (包含其他浮点数) 的数量: {non_binary_count}")
    
    if non_binary_count == 0:
        print("\n🎉 结论：目前小数据集中的全部数据的输出向量 100% 是二值化的！")
    else:
        print("\n⚠️ 结论：存在非二值化输出，说明网络并没有彻底强硬截断所有特征。")
    print("="*50)

if __name__ == "__main__":
    main()