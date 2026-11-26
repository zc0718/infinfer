Third-party notices
===================

This package **does not redistribute third-party source code**. Every dependency listed in
`metadata.json` / `conandata.yml` is consumed as its own Conan package (or, for the test
framework, as a build-only package), so its own licence files travel with that package
rather than with this one. The machine-readable record of what each dependency is, which
licence we rely on, and where that was read from is `.github/project/deps.json`, checked by
`.github/project/deps_gate.py` on every gate run.

The `NOTICE` file next to this one is template text: it states the licence of *this*
project (Apache-2.0 for software, CC BY-NC 4.0 for documentation) and is pinned by
`.github/project/scaffold_gate.py` rule G5 so that it cannot be rewritten by accident.

If this project ever vendors third-party code into the tree, the attribution for it goes in
the section below — that is the only thing `deps_gate.py` rule D4 reads, so a component we
ship cannot arrive without a notice line appearing here, and a component we do not ship
cannot appear here by mistake.

<!-- redistributed:begin -->
<!-- redistributed:end -->

Inventory (not redistributed, for reference)
--------------------------------------------

| dependency | licence we rely on | used from |
|---|---|---|
| zlib | Zlib | `src/ctest.c` |
| pcre2 | BSD-3-Clause | `src/ctest.c` |
| Eigen3 | MPL-2.0 | `src/cpptest.cpp` |
| etl | MIT | `src/etl.cpp` |
| GTest | BSD-3-Clause | `test_package/test/unit/` |
| pybind11 | BSD-3-Clause (unverified on this machine: bindings are disabled) | `api/python_bindings.cpp` |

---

第三方声明（中文摘要）
----------------------

本包**不以源码形式重分发任何第三方代码**：每个依赖都是独立的 Conan 包，其许可证文件随
该包分发，而不随本包。依赖清单、我们依据的许可证标识以及读取证据，全部记录在
`.github/project/deps.json`，由 `.github/project/deps_gate.py` 在每次门禁运行时核对。

同目录的 `NOTICE` 是模板文本，声明**本项目自身**的许可（软件 Apache-2.0、文档 CC BY-NC
4.0），并由 `scaffold_gate.py` 的 G5 规则按 sha256 钉住，防止被无意改写。

若将来把第三方代码 vendor 进本仓，归因行写在上方 `<!-- redistributed:begin -->` 与
`<!-- redistributed:end -->` 之间——那是 `deps_gate.py` D4 唯一读取的区域，于是「我们分发
的组件」不可能不出现归因行，而「我们不分发的组件」也不会被误写进去。
