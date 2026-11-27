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
GATES="docannot_gate.py census_gate.py registry_gate.py isa_gate.py apicontract_gate.py rootlayout_gate.py scope_gate.py toolchain_gate.py initform_gate.py sanitize_gate.py header_dialect_gate.py coverage_gate.py procdocs_gate.py claim_gate.py gaps_gate.py deps_gate.py docvoice_gate.py scaffold_gate.py ci_policy_gate.py compliance_gate.py p3_gate.py vendor_gate.py"
for gate in ${GATES}; do
    echo "--- .github/project/${gate}"
    python3 ".github/project/${gate}" || status=1
done

# The gates' own proofs. A gate that cannot be shown to fail is a gate nobody should trust, and its
# fixture is the thing that breaks silently when a rule's wording changes -- measured: adding the
# deps licence rule turned four of its fixtures red, and nothing noticed, because `--selftest` was
# only ever run by hand. All 21 run in under two seconds.
# 门禁自己的证明。一条证明不了会红的闸不值得信任，而规则的措辞一改，坏掉的正是它的 fixture——
# 实测：给 `deps` 加许可规则后它的四条 fixture 变红，而没人发现，因为 `--selftest` 只被手工跑过。
# 21 条合计不到两秒。
for gate in ${GATES}; do
    echo "--- .github/project/${gate} --selftest"
    python3 ".github/project/${gate}" --selftest || status=1
done

if [ "${status}" -ne 0 ]; then
    echo "::error::one or more project-owned gates failed"
fi
exit "${status}"
