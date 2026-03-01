import torch
from flx.setup.experiments import EXTRACTORS
from flx.setup.datasets import get_nist14

def main():
    # 1. 加载模型
    print("正在加载训练好的 DeepPrint_Tex_512 模型...")
    extractor = EXTRACTORS["DeepPrint_Tex_512"].load()
    model = extractor.model
    model.eval()
    model.to("cpu")

    # 2. 读取 NIST14 真实数据集
    print("正在加载 NIST14 真实数据集...")
    nist14_path = "/root/autodl-tmp/fixed-length-fingerprint-extractors/data/fingerprints/NIST14"
    dataset = get_nist14(nist14_path)

    # 3. 获取数据集里的第一张指纹图片
    first_id = dataset.ids[0]
    real_image = dataset.get(first_id)
    input_tensor = real_image.unsqueeze(0).to("cpu")
    print(f"成功读取第一张真实图片，输入张量形状为: {input_tensor.shape}")

    # 4. 进行前向传播，提取特征
    print("正在提取特征...")
    with torch.no_grad():
        output = model(input_tensor)

    # 5. 精准提取我们要的纹理特征
    feature_tensor = output.texture_embeddings

    # 6. 打印结果，揭晓答案！
    print("\n" + "="*50)
    print(f"提取出的张量形状 (Shape): {feature_tensor.shape}")
    print(f"数据类型 (Dtype): {feature_tensor.dtype}")
    
    print("\n这张真实指纹提取出的前 30 个特征数值:")
    # 修复：因为特征已经是 1 维张量了，直接取前 30 个元素即可
    values = feature_tensor[:30].numpy()
    print(["{:.4f}".format(v) for v in values])
    print("="*50)

if __name__ == "__main__":
    main()