#!/usr/bin/env bash
# Project-owned gates, called by the template's `project-gates` job.
#
# The template reserves this one hook on purpose: a derived project adds its own
# checks here instead of editing a template workflow, so the next template
# upgrade does not have to reconcile a hand-merged job list. The job runs this
# file when it exists and prints a notice when it does not, which is why a gate
# that is only listed in a local note is a gate that never runs.
#
# Every gate prints its own verdict and exits non-zero when red; this script runs
# them all and fails if any of them failed, so one red gate does not hide the
# verdict of the next.
#
# 模板预留的项目门禁钩子。派生项目在这里加自己的检查，而不是去改模板
# workflow —— 这样模板升级时不必再手工合并 job 列表。钩子存在时执行它，不存在时
# 只打印一条 notice，所以"只写在本地笔记里的门禁"等于没跑。

set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

# docannot and census read YAML; the hook has no step of its own to install it.
if ! python3 -c "import yaml" >/dev/null 2>&1; then
    python3 -m pip install --quiet pyyaml || {
        echo "::error::ci.sh could not install pyyaml, so the YAML-reading gates cannot run"
        exit 1
    }
fi

status=0
for gate in docannot_gate.py census_gate.py registry_gate.py isa_gate.py apicontract_gate.py rootlayout_gate.py scope_gate.py toolchain_gate.py initform_gate.py sanitize_gate.py header_dialect_gate.py coverage_gate.py procdocs_gate.py claim_gate.py gaps_gate.py; do
    echo "--- .github/project/${gate}"
    python3 ".github/project/${gate}" || status=1
done

if [ "${status}" -ne 0 ]; then
    echo "::error::one or more project-owned gates failed"
fi
exit "${status}"
