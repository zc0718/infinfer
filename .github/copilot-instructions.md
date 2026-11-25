# infinfer — agent 工作须知

- 先读 `workspace/KICKOFF.md`，再读 `workspace/PLAN-2026-11-25.md` §2（不变量）与 §8（执行规约）。
- 每个任务完成前必须：`python3 .github/misc/gates.py all` 绿 + 新门禁能演示变红 + `gaps.md` 增行。
- 硬禁止：`include/` 下出现架构头或厂商头；kernel 签名出现 shape/stride；用 `#define` 裁剪算子；新增顶层目录。
