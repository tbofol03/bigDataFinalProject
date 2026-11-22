# 仓库合并说明

这个版本包含数据清洗、选择性模型调用、缓存、并发与限流控制、结果校验，
以及三种规模的实验数据和评分。目录和说明使用通用项目名称。

## 合并步骤

1. 在现有仓库中更新本地代码，然后新建分支。
2. 将交接包中的文件按目录合并到仓库根目录。同名源码和 README 是本次更新内容；
   如果同学已经修改了同一文件，先比较差异再合并。保留其他已有工作。
3. 在仓库根目录运行下面的离线检查。已有结果不需要再付费调用 API。

```bash
git switch -c improve-cleaning-pipeline
python -m unittest discover -s tests -v
python finalize_results.py
```

预期：7 项测试通过；汇总输出 `all_requests_completed: true`、
`remaining_request_errors: 0`。三种规模为 500、5,000、12,421 条。

4. 检查暂存内容，提交分支并创建 Pull Request。

```bash
git add README.md HANDOFF.md requirements.txt .gitignore *.py tests data inputs runs results supporting MANIFEST.sha256
git diff --cached --stat
git commit -m "Improve cleaning pipeline and add reproducible experiments"
git push -u origin improve-cleaning-pipeline
```

建议 PR 标题：`Improve cleaning pipeline and add reproducible experiments`

建议 PR 描述：

> Adds selective LLM routing, source-evidence validation, response caching,
> concurrent requests and shared rate-limit handling. Preserves rule outputs
> when requests fail or proposals are rejected. Includes frozen XML/manifests,
> measured 500/5,000/12,421-record runs and reproducible source-based scores.
>
> Validation: 7 unit tests passed; offline result assembly completed with
> zero remaining request errors. Full-workload nonempty address formatting
> agreement is 500/506 (98.81%), versus 466/506 (92.09%) for the original rules.

## 文件用途

- 根目录 Python 文件、`tests/`：代码及测试；所有配套模块应一并合并。
- `data/`、`inputs/`：固定 XML 快照、抽样清单和参考数据。
- `runs/`：原始运行与补跑结果；保留两者以核对实际请求数、耗时和成本。
- `results/`：评分及三种规模的汇总。
- `supporting/`：增强规则及候选重复记录的对照数据。
- `MANIFEST.sha256`：交接时的文件校验值；它记录此交接版本，修改后的文件会产生不同哈希。

交接包未包含 API 密钥、响应缓存、虚拟环境或临时文件。数据集中的建筑名称
是实验字段，按原始内容保留。姓名、学号、课程编号等人员或课程信息不属于此包。

## 结果表述

格式一致率基于同一批 300 条 XML 参考记录，其中 506 个地址字段非空。
它衡量与参考格式的一致性；外部地址准确性未经验证。增强规则基线的
非空地址一致率为 504/506，因此不能据此声称模型优于全部规则方案。
不同规模的并发配置不同，耗时应按实际运行过程解释。
