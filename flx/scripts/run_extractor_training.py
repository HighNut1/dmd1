import torch
from flx.setup.experiments import TESTSETS, EXTRACTORS
from flx.setup.datasets import get_nist14
from flx.data.dataset import Dataset, ConstantDataLoader
from flx.data.label_index import LabelIndex

# 选择纹理分支，数字代表你要提取的二值向量长度（例如 512 维）
# 你可以在 experiments.py 的 EXTRACTORS 字典里查看支持的维度，如 DeepPrint_Tex_256, DeepPrint_Tex_1024 等
EXTRACTOR = EXTRACTORS["DeepPrint_Tex_512"] 
NUM_EPOCHS = 75

def main():
    # 1. 加载端到端二值化改造后的模型
    extractor = EXTRACTOR.load()
    output_dir = EXTRACTOR.get_dir()
    
    print(f"准备训练模型，输出目录: {output_dir}")

    # 2. 读取 NIST14 数据集 (直接指向你的绝对路径)
    nist14_path = "/root/autodl-tmp/fixed-length-fingerprint-extractors/data/fingerprints/NIST14"
    fingerprints = get_nist14(nist14_path)
    print(f"成功加载 NIST14 数据集: 包含 {fingerprints.num_subjects} 个 Subject")

    # 3. 自动生成网络分类需要的 Labels (0~1999)
    labels = Dataset(LabelIndex(fingerprints.ids), fingerprints.ids)
    
    # 4. 填充空的 Minutia Maps
    # 因为 NIST14 没有细节点标注，我们用空的 Tensor 占位，DeepPrint_Tex 分支在训练时也会自动忽略它
    minutia_maps = Dataset(ConstantDataLoader(torch.tensor([])), fingerprints.ids)

    # 5. 开始拟合训练
    # 这里为了专注于 NIST14 的训练，我们将额外的验证集设为 None 
    # (模型会在训练集上通过 Center Loss 和 CrossEntropy 学习如何输出 {-1, 1})
    extractor.fit(
        fingerprints=fingerprints,
        minutia_maps=minutia_maps,
        labels=labels,
        validation_fingerprints=None,
        validation_benchmark=None,
        num_epochs=NUM_EPOCHS,
        out_dir=output_dir,
    )

if __name__ == "__main__":
    main()