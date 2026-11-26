# Local Conan recipes（本地配方目录）

这里放**我们自己维护的** Conan recipe：Conan Center 没有的库，或 CCI 版本落后到不满足需要的库
（判定规则见 `workspace/PLAN-2026-11-25.md` §3.9 ADR-006）。

This directory holds the Conan recipes **we maintain ourselves**: libraries Conan Center does not
carry, or whose CCI version is too old for what we need (the rule is ADR-006 in §3.9 of the plan).

## 这里的文件不进版本库 / Nothing here is committed

`.gitignore` 只放行本文件，其余内容一律不入库：

```gitignore
/workspace/conan/recipes/*
!/workspace/conan/recipes/README.md
```

原因（PLAN 416 行、1004 行硬禁止）：配方里会写下载地址与 `sha256`，散在仓库外才是它该待的地方；
分发走私有 remote `hetai`，`conandata.yml` 里只留**需求行 + 版本**。

Why: recipes carry download URLs and `sha256` pins, and the plan forbids moving them into the
version library; distribution goes through the private `hetai` remote, while `conandata.yml` keeps
only the requirement line and the version.

## 加一份新配方时的顺序 / Order of work when adding one

1. 在这里建 `<name>/config.yml` + `<name>/all/{conanfile.py,conandata.yml,test_package/}`（CCI 版式）；
2. `conan export` 到本地 cache，确认 `conan create` 能过；
3. 版本与 `sha256` 钉死；`test_package/` 里真的 include 一次并使用一个符号；
4. 把**需求行**加进 `conandata.yml`，并在 `.github/project/deps.json` 里登记它（licence、`used_from`、
   `attribution`）——`deps_gate` 会核对这两处是否一致；
5. 若它要成为可选后端，`conanfile.py` 需要 `with_<name>` option：**那是脚手架文件，改动须先取得业主同意**
   （gaps G-36），且按 PLAN P0.3 的做法加条件 `self.requires(..., transitive=False)`。

## 当前状态 / Current state

空。P0.7（配方环境就位）仍是待办：它的验收是「`conan export` 本地可用 + 关掉 option 时干净 clone 仍能
构建（INV-6）」，而这两条都要**至少一份配方**才能验。今天没有待引入的厂商库，所以这项挂着等第一个真实
依赖（见台账 G-88）。

Empty. P0.7 is still open: its acceptance needs at least one recipe. Nothing is due to be introduced
yet, so it waits for the first real dependency (gaps G-88).
