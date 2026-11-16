# 2026-11-05 benchmark 增量分析

用户已授权启动，并指定新 SSH 端口 40665。本目录是新增分析的代码；服务器产物保留在服务器。

服务器目录：`/liaozizhuo/hypoblast/benchmarks/hypo-benchmark-extension-20261105`。

## 范围与固定条件

- 原表达 benchmark：`hypo-forward-embedding-combo-20261029`，24 个已有结果行，仅作为冻结参考。
- 新增 PerturbGen：43、44、45 三个训练种子，共 3 个新表达结果行；最终表达比较为 11 种方法、27 行。旧训练不重跑。
- 延用 146/21/14 条件划分、2,000 HVG 顺序、原始表达尺度和既有 10 类 V7 注释。
- 比例分析另行导出旧模型的逐细胞预测，通过共同注释器汇总；这不会改变旧表达结果。
- 当前测试条件曾参与项目开发，按 development-exposed 测试集报告。

## PerturbGen 实现与运行

官方源码：[Lotfollahi-lab/Perturbgen](https://github.com/Lotfollahi-lab/Perturbgen)，固定提交 `2507b90f52613384da258eae45a4b01d89f2f7aa`。

官方编码器来自 [Hugging Face](https://huggingface.co/lotfollahi-lab/PerturbGen/tree/60aef6435163d8dfb87513a9b8dad9ce771e3c6a)，SHA-256 为 `49bfc4283ca82ba39f5082920f7f1d962cd6ae768dae586731ffac87fd75821b`，本地获取后上传，服务器再次核对。

使用官方冻结 scMaskGit 编码器、六层 PerturbGen 解码器、MaskGIT 目标与 CountHead 表达预测头。无序药物 embedding 的求和与药物数投影组成一个额外 cross-attention memory token。这是 **PerturbGen 的终点对照／药物组合适配版本**，不宣称复现原论文的时间轨迹任务。

仅有的兼容修正：可选 Geneformer EmbExtractor 导入改为缺失时显式报错的惰性依赖；BF16 推理 logits 在原生 MaskGIT 掩码前转 float32，避免掩码常数溢出。原生核心网络与生成算法保留。

1. `prepare_tokens.py`：对允许的训练、验证和对照细胞进行官方词表／中位数归一化的 gene-rank tokenization。源序列最长 4096，目标最长 2048；测试目标行保持为零。
2. `train_perturbgen.py`：训练 mask decoder 与组合适配；每条件每轮随机抽样 32 个细胞，microbatch 8、累积 4 次。中间 checkpoint 按固定 21 个验证条件的 mask loss 选择。
3. `finish_perturbgen.py`：冻结 mask 模型，训练原生 CountHead。每个训练条件最多 5000 个固定抽样细胞缓存 teacher embeddings。验证使用所有 21 个条件的全部匹配对照细胞，完全从源细胞生成。表达头按验证 PCC_gene、比例头按验证 MAE 分别选择。两阶段均最多 3000 轮、patience 20；各阶段预算如实单列，不声称所有模型计算量相同。
4. 表达测试沿用旧 evaluator、全部对照源细胞；比例主分析使用 5000 个生成细胞 × 抽样种子 101/102/103。冻结表达 checkpoint 的比例结果与按比例重新选出的 checkpoint 分开保存，不能混在一个统一选模的排行榜中。
5. `run_perturbgen.py --execute`：仅启动上述 3 个新行；每个种子在 mask 阶段完成后自动进入 CountHead 与导出。没有额外并发上限。

用户随后开放 34272 上四张 A100 80GB，与 40665 共享 `/liaozizhuo`。43、45 从保存的 mask checkpoint 迁移到 34272 的 GPU 0、1，44 在 40665 的 GPU 1 保持原进程。剩余 GPU 暂未分配；不为占满硬件而增加原计划以外的实验。

迁移恢复模型、AdamW 和 scheduler 状态。旧 checkpoint 未包含随机数状态，按固定的 8 × 4 原始采样／原生 mask draw 流重放到 checkpoint 轮末，再从下一轮继续；未保存的当前轮重算。新的 `mask_last.pt` 每轮保存完整 RNG，`mask_best.pt` 仍按原验证标准选择。`training_state.py` 的定向测试检查恢复后下一优化步及随机采样流一致。

状态：`audit/perturbgen_dispatch.json`、`audit/perturbgen_queue_<hostname>.json`、`runs/perturbgen_seed*/status.json`；日志：`logs/perturbgen_seed*_{mask,finish}.log`。旧 `perturbgen_queue.json` 由 dispatch 文件取代。`MASK_COMPLETE.json` 只代表第一阶段完成；只有 `COMPLETE.json` 才代表该种子的最终测试导出完成。

## 已启动的比例与加性分析

`prepare_composition.py` 从 `metadata_updated_v7.csv` 重算比例并核对冻结 truth，训练共同的多项 logistic 注释器：按条件平衡抽取训练细胞，C 仅由验证集选定。旧 raw-h5ad 比例表的部分细胞类型标签与 V7 不一致，不混用。

`export_composition.py` 对 CPA、chemCPA、STATE、G2CP、PerturbNet、CellFlow 的 18 个冻结 checkpoint 做逐细胞推理；每条件 5000 个细胞 × 3 次抽样。主要比例是逐细胞分类概率的平均，hard label 计数作为诊断保存。

`analyze_additivity.py` 计算描述性偏离：

- 原始参考：`p0 + Σ(pi − p0)`；V7 核对后，5 条件 × 10 类中有 10 个数值越界。
- 主参考：`softmax(clr(p0) + Σ(clr(pi) − clr(p0)))`，伪计数 1e-4；另算 1e-5、1e-3 敏感性及欧氏单纯形投影参考。
- 同批次、训练集单药齐全的候选条件为 H31、H46、H71、H85、H132；其余 9 个不用于严格加性比较。
- 三阶包含排除分解仅对训练集中单药与双药齐全的 H85、H132 计算。H71 缺少训练内双药参考，排除。
- 药物身份和批次已匹配；剂量、时间匹配仍待逐项核验。因此这里是指定参考下的计算偏离，尚不能称为生物学协同或拮抗。没有用单细胞充当独立生物重复做显著性检验。

关键服务器产物：`results/metrics_available.csv`、`results/additivity_descriptive.csv`、`audit/additivity_complete.json`、`audit/annotation_reconciliation.json`。

## 定向检查

比例计算的 3 个单元测试覆盖概率合法性、标签对齐、JSD、CLR、投影和并列排名。服务器另行检查原生 PerturbGen 的梯度更新、药物顺序不变性、源序列生成、编码器冻结及原生 CountHead 梯度。最终报告须区分“正在训练”和“已有最终结果”，不可把 smoke loss 或中间验证 loss 当作测试 benchmark 分数。
