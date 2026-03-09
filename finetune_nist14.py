import os
import torch
import torch.nn.functional as F
import random
from torch.optim import Adam
from tqdm import tqdm

from flx.setup.experiments import EXTRACTORS
from flx.setup.datasets import get_nist14
from flx.data.dataset import Identifier

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载官方满级预训练模型
    print("正在加载官方 DeepPrint_TexMinu_512 预训练模型作为基座...")
    extractor = EXTRACTORS["DeepPrint_TexMinu_512"].load()
    model = extractor.model
    model.to(device)
    model.train() # 开启训练模式

    # 2. 设置优化器与损失函数
    # 秘诀1：使用极小的学习率 (1e-5)，防止破坏已经学到的世界级通用特征
    optimizer = Adam(model.parameters(), lr=1e-5)

    # 秘诀2：使用 Triplet Loss (Margin=0.5 意味着正样本距离必须比负样本近 0.5 以上)
    triplet_loss = torch.nn.TripletMarginLoss(margin=0.5, p=2)

    # 3. 加载 NIST14 数据集
    print("正在初始化 NIST14 数据集...")
    nist14_path = "/root/autodl-tmp/fixed-length-fingerprint-extractors/data/fingerprints/NIST14"
    dataset = get_nist14(nist14_path)

    # NIST14 有 2000 个 Subject，每个有 F(编号0) 和 S(编号1) 两张图
    # 我们取出所有存在的 Subject ID (过滤掉之前发现的损坏图片)
    valid_subjects = []
    for sub in range(2000):
        try:
            # 试探性加载，如果图坏了就跳过
            _ = dataset.get(Identifier(sub, 0))
            _ = dataset.get(Identifier(sub, 1))
            valid_subjects.append(sub)
        except:
            continue
            
    print(f"有效 Subject 数量: {len(valid_subjects)}")

    epochs = 10
    batch_size = 32 # 你的 4090 显存很大，可以用 32 甚至 64

    print("\n🚀 开始使用三元组损失 (Triplet Loss) 进行微调...")

    for epoch in range(epochs):
        random.shuffle(valid_subjects)
        epoch_loss = 0.0
        batches = 0

        # 添加进度条
        pbar = tqdm(range(0, len(valid_subjects), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for i in pbar:
            batch_subs = valid_subjects[i:i+batch_size]
            if len(batch_subs) < 2:
                continue

            anchor_imgs = []
            positive_imgs = []

            for sub in batch_subs:
                # 获取 Anchor(图F) 和 Positive(图S)
                anch_img = dataset.get(Identifier(sub, 0)).unsqueeze(0)
                pos_img = dataset.get(Identifier(sub, 1)).unsqueeze(0)
                anchor_imgs.append(anch_img)
                positive_imgs.append(pos_img)

            anchor_tensor = torch.cat(anchor_imgs, dim=0).to(device)
            positive_tensor = torch.cat(positive_imgs, dim=0).to(device)

            # 前向传播提取特征
            optimizer.zero_grad()
            out_a = model(anchor_tensor)
            out_p = model(positive_tensor)

            # 拼接 512 维特征，由于现在是批量运算，拼接维度改为 dim=1
            feat_a = torch.cat([out_a.texture_embeddings, out_a.minutia_embeddings], dim=1)
            feat_p = torch.cat([out_p.texture_embeddings, out_p.minutia_embeddings], dim=1)

            # 对特征进行 L2 归一化 (度量学习标准操作)
            feat_a = F.normalize(feat_a, p=2, dim=1)
            feat_p = F.normalize(feat_p, p=2, dim=1)

            # 秘诀3：巧妙构建负样本！把 batch 里的 Anchor 特征往下滚动一位
            # 这样每个人对应的负样本，就是同 Batch 里的另一个人
            feat_n = torch.roll(feat_a, shifts=1, dims=0)

            # 计算 Triplet Loss 并反向传播
            loss = triplet_loss(feat_a, feat_p, feat_n)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_loss = epoch_loss / batches
        print(f"👉 Epoch [{epoch+1}/{epochs}] 平均损失: {avg_loss:.4f}")

    # 4. 保存微调后的专属模型
    save_dir = "/root/autodl-tmp/fixed-length-fingerprint-extractors/models/DeepPrint_TexMinu_512"
    save_path = os.path.join(save_dir, "finetuned_nist14_model.pyt")
    
    checkpoint = {
        "model_state_dict": model.state_dict()
    }
    torch.save(checkpoint, save_path)
    print(f"\n🎉 微调彻底完成！专属模型已保存至:\n{save_path}")

if __name__ == "__main__":
    main()