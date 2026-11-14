# A051 可运行项目验收记录

本文件是项目实施与验收清单，不是课程项目报告。项目报告等待用户另行明确要求后撰写。

组号 A051；成员李石屹（1320241091）独立完成论文复现、实验、接口、部署及提交。

依据：课程《语音识别与合成课程大作业.pptx》及用户提供的要求截图；提交要求以 `lym-lab/algorithm_research_frontend` 的算法提交指南、接入规范、小组 README 模板及自查清单为准。

|评分项目|验收条件|本轮状态与证据|
|---|---|---|
|接入系统前端（10）|方法可检索、详情及真实上传/参数/处理/预览/下载通过|本地实际浏览器完成 NMR256 嵌入、下载、重上传提取，BER=0；`outputs/browser_validation/result.json` 与嵌入/提取截图；云端浏览器复核已通过，见云端验收记录|
|论文核心方法（25）|STFT、嵌入、波形重分析提取、MSE/NMR、两种消息长度及训练日程有代码与证据|60 候选 best/last/result/history 及四模型选择已通过 `scripts.verify_final_training`；见 `outputs/final_training_audit.json`；严格一致性差异仍按 `audit-20261021.md` 披露|
|实验结果验证（10）|四模型同一 RWC 3000 段、BER/SNR/NMR、真实 PEAQ、合理对齐与统计|全部真实执行完成；ODG 为四模型共同有效2970段，相同30段未定义结果保留 NaN 与逐样本诊断；`outputs/final_analysis/comparison.csv`、`peaq_cohort.json` 及四模型逐段 CSV；2026-11-01 再核对汇总与原始表一致|
|案例可视化与分析（10）|正式模型真实处理、波形/差值、谱图/掩蔽案例、实测模型对比|四模型 HTTP 嵌入、下载、重上传独立提取已验证；`outputs/case/mask_noise.png`、`case.json`、`outputs/final_analysis/` 箱线图与 `outputs/service_samples/` 四模型真实演示结果|
|华为云整套模型及功能（10）|云主机四权重校验、服务健康、前端全流程、重启后可用|2026-11-01 香港实例四模型完整权重包SHA256通过；公网HTTPS四模型嵌入/下载/重上传提取及整机重启后同样四模型复测均通过；实际浏览器NMR256下载再上传提取BER=0。见 `outputs/cloud_validation_20261101/`、`outputs/cloud_reboot_validation_20261101/`，截图本地留存；根README含公开入口与权重链接|
|README（5）|根目录按模板写明成员、实现、环境、参数、真实样例、结果、局限及所有附加文档入口|Fork根README已重写且补齐真实云端、权重下载与证据入口；分支 `group_a051_submission`；最终PR完成后登记实际链接|
|项目报告（30）|由用户另行启动|本轮不撰写、不生成、不修改报告|

## 实验与论文差异的处理

训练按已冻结且已执行完成的配置验收。单位化 STFT/MSE 尺度、191500 bps 筛选容差、U 矩阵边界、验证集选模及图文歧义均在旧审计中有证据；不因课程功能可运行就声称逐点等同作者未公开实现或达到论文数值。60 候选完成不等于四模型客观评价及云部署完成。

PEAQ 初始失败样本 index=109 的参考音频 RMS 约 2.51e-5。Kabal 输出 `PEAQ Data Boundaries: 0 - 0`，零有效帧导致 `Total NMRB: NaN` 和 ODG NaN。诊断见 `outputs/evaluation/MSE256/peaq.csv.undefined_109.txt`。参考源码保持不变；包装器逐行记录 `ok` 或 `undefined_reference_output` 并保存原始诊断。真正的运行异常仍报错；不会拿任意分数替代未定义结果。

真实 MUSHRA 听评需要真实参与者，自动化不能生成有效的人类评分。本轮准备正式模型的七段听评材料与收集工具；未组织听评就不声称复现主观结论。课程截图未单独列出必须招募20名听者。

## 当前执行归属

本对话负责继续项目、验证、部署与提交。训练已经完成，保留所有原始训练检查点。GitHub 目标是用户本次明确指定的新 Fork，不重建旧 `A051` 目录。
