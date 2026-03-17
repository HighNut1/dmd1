import os
import glob
import torch
from torchvision import transforms
import cv2
import numpy as np

# 导入项目中定义的模型架构
from flx.models.deep_print_arch import DeepPrint_TexMinu

def check_is_binary(tensor):
    """
    检查张量中的所有元素是否只包含 1.0 和 -1.0
    (考虑到 sign(0) = 0 的极小概率情况，我们也可以包容 0，但通常是 1 和 -1)
    """
    # 检查是否全部是 1, -1 或 0
    valid_values = (tensor == 1.0) | (tensor == -1.0) | (tensor == 0.0)
    is_strictly_binary = torch.all(valid_values).item()
    
    # 获取张量中实际存在的唯一值，用于打印诊断
    unique_vals = torch.unique(tensor).tolist()
    return is_strictly_binary, unique_vals

def load_and_preprocess_image(image_path):
    """
    按照项目标准加载并预处理图像
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)), # DeepPrint 要求的输入尺寸
        transforms.ToTensor()
    ])
    
    img_tensor = transform(img)
    # 增加 batch 维度: [1, 1, 299, 299]
    img_tensor = img_tensor.unsqueeze(0) 
    return img_tensor

def main():
    # 1. 配置参数
    dataset_dir = "notebooks/example-dataset/" # 可以替换为您的 NIST14 数据集路径
    texture_dims = 256
    minutia_dims = 256
    
    # 2. 初始化模型并设置为评估模式
    print("正在初始化 DeepPrint_TexMinu 模型...")
    # num_fingerprints 在提取特征时并不重要，随便设一个即可
    model = DeepPrint_TexMinu(num_fingerprints=1000, 
                              texture_embedding_dims=texture_dims, 
                              minutia_embedding_dims=minutia_dims)
    model.eval() # 必须设置为 eval 模式
    
    # 注意：这里我们甚至不需要加载预训练权重。
    # 因为 STE(Sign) 函数位于网络架构最末端，即使是随机初始化的权重，
    # 经过 sign 函数后也必须是二值化的。如果您想加载权重，可以取消下面的注释：
    # model_path = "path/to/your/best_model.pt"
    # model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # 3. 收集所有图像
    print(f"正在扫描数据集目录: {dataset_dir}")
    image_paths = glob.glob(os.path.join(dataset_dir, "**", "*.png"), recursive=True)
    if not image_paths:
        print("未找到任何 .png 图像，请检查路径。")
        return

    print(f"共找到 {len(image_paths)} 张图像。开始二值化检测...\n")

    total_images = 0
    failed_images = 0

    # 4. 遍历检测
    with torch.no_grad():
        for path in image_paths:
            img_tensor = load_and_preprocess_image(path)
            if img_tensor is None:
                continue
            
            # 前向传播提取特征
            output = model(img_tensor)
            tex_emb = output.texture_embeddings
            minu_emb = output.minutia_embeddings
            
            # 拼接为一个完整的特征向量
            combined_emb = torch.cat((tex_emb, minu_emb), dim=-1)
            
            # 检测数值
            is_binary, unique_vals = check_is_binary(combined_emb)
            
            total_images += 1
            if not is_binary:
                failed_images += 1
                print(f"[失败] 图像 {path} 输出了非二值化数值! 包含的数值: {unique_vals}")
            
            # 每处理 50 张打印一次进度
            if total_images % 50 == 0:
                print(f"已检测 {total_images}/{len(image_paths)} 张...")

    # 5. 输出总结
    print("-" * 40)
    print("检测完成!")
    print(f"总计检测图像数: {total_images}")
    if failed_images == 0:
        print("🎉 测试通过！所有数据输出的特征向量均为严格的二值化向量 (只包含 1.0 和 -1.0)。")
        
        # 顺便打印一下特征维度和占用的字节数作为参考
        print(f"特征总维度: {combined_emb.shape[-1]} 维")
        print("提示：在后续比对代码中，您可以将它们转换为 uint8 类型以使用汉明距离加速！")
    else:
        print(f"❌ 测试失败！发现 {failed_images} 张图像输出了包含连续浮点数的向量。请检查 STE 代码是否生效。")

if __name__ == "__main__":
    main()