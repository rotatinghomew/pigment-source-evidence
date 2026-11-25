"""命令行入口：建案、重开与报告输出。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze import EXCLUDED, INVOLVED_ALL, INVOLVED_SOME
from .casefile import open_case, reopen_case
from .model import load_records
from .analyze import analyze_case
from .isolation import plan_isolation, schedule_inquiries

CLASS_LABEL = {
    INVOLVED_ALL: '所有解释均涉入（必停/必通知）',
    INVOLVED_SOME: '部分解释涉入（待检）',
    EXCLUDED: '现有资料排除（有条件放行，需签认）',
}


def build_report(data_dir: str | Path) -> str:
    batches, scales, jobs, retains, slots, inquiries, empties = load_records(data_dir)
    analysis = analyze_case(batches, scales, jobs, retains, empties)
    plan = plan_isolation(analysis, slots)
    schedule = schedule_inquiries(analysis, inquiries, scales)

    lines: list[str] = []
    lines.append('# 色漂批源研判与隔离单')
    lines.append('')
    lines.append('## 1. 可行的补称场景')
    for result in analysis.scenarios:
        sc = result.scenario
        if not result.feasible:
            status = '物理不可行（时间/用量约束冲突，排除）'
        elif not result.evidence_feasible:
            status = '物理可行但与留样证据冲突（排除）'
        else:
            status = '物理与留样证据均相容'
        note = f'；{sc.note}' if sc.note else ''
        lines.append(f'- {sc.name}: {status}{note}')
        if result.feasible:
            lines.append(
                f'  相容异常留样: {", ".join(result.anomalous_retains) or "无"}'
            )
    lines.append('')

    lines.append('## 2. 当前异常与候选批相容性')
    for jid in analysis.anomalous_jobs:
        compatible = analysis.anomaly_compatible_batches[jid]
        retain_ids = [r.id for r in retains if r.job == jid]
        lines.append(
            f'- 工件 {jid}（留样 {", ".join(retain_ids)}）：'
            f'相容高漂批 {sorted(compatible) or "无，资料冲突"}'
        )
    lines.append('')
    lines.append('### 共同解释（最小充分批组）')
    for group in analysis.minimal_groups:
        lines.append(f'- {sorted(group)}')
    lines.append(
        f'- 跨所有解释均涉入: {sorted(analysis.common_culprits) or "无"}'
    )
    lines.append(
        f'- 现有资料排除的高漂批: {sorted(analysis.ruled_out_batches) or "无"}'
    )
    lines.append('')

    lines.append('## 3. 工件三分类')
    lines.append(
        '| 工件 | 状态 | 分类 | 必然批源 | 可能批源 | 物理包络色值 | 证据相容色值 | 依据 |'
    )
    lines.append('|---|---|---|---|---|---|---|---|')
    for jid in sorted(analysis.verdicts):
        v = analysis.verdicts[jid]
        lines.append(
            f'| {jid} | {v.state} | {CLASS_LABEL[v.classification]} | '
            f'{", ".join(sorted(v.necessary_batches)) or "—"} | '
            f'{", ".join(sorted(v.possible_batches)) or "—"} | '
            f'{v.achievable_range[0]:.3f}–{v.achievable_range[1]:.3f} | '
            f'{v.evidence_range[0]:.3f}–{v.evidence_range[1]:.3f} | '
            f'{"；".join(v.reasons)} |'
        )
    lines.append('')

    lines.append('## 4. 隔离与通知清单')
    lines.append('### 已发件（通知待审，不占隔离位）')
    if plan.notify:
        lines.extend(
            f'- {jid}: {CLASS_LABEL[analysis.verdicts[jid].classification]}'
            for jid in plan.notify
        )
    else:
        lines.append('- 无')
    lines.append('')
    lines.append('### 在制件停用/待检安置')
    for slot_id, occupants in sorted(plan.slot_usage.items()):
        used = sum(analysis.verdicts[j].weight for j in occupants if j in analysis.verdicts)
        lines.append(f'- 隔离位 {slot_id}（已用 {used:g}kg）: {", ".join(occupants)}')
    if plan.unplaced:
        lines.append('### 本班无法保障（保留待检，不得作出货结论）')
        for jid, reason in sorted(plan.unplaced.items()):
            lines.append(f'- {jid}: {reason}')
    lines.append('### 有条件放行（须主管签认，非无条件出货）')
    if plan.conditional_release:
        lines.extend(f'- {jid}' for jid in plan.conditional_release)
    else:
        lines.append('- 无')
    lines.append('')

    lines.append('## 5. 补证排程（按可排除的冲突解释排序）')
    for opt in schedule.selected:
        q = opt.inquiry
        lines.append(
            f'- {q.id}（{q.kind} -> {q.target}，班组 {q.crew}，'
            f'窗 {q.start:g}–{q.end:g}，价值 {opt.value:g}）: {opt.rationale}'
        )
    if schedule.conflicts:
        lines.append('### 互斥关系')
        lines.extend(f'- {text}' for text in schedule.conflicts)
    lines.append('')
    lines.append('### 备选班组组合')
    for combo, value in schedule.alternative_sets:
        lines.append(f'- {list(combo)}: 合计价值 {value:g}')
    lines.append('')

    lines.append('## 6. 下一步所缺资料')
    missing: list[str] = []
    for jid, reason in plan.unplaced.items():
        missing.append(f'隔离资源（{jid}：{reason}）')
    for opt in schedule.selected:
        missing.append(f'执行 {opt.inquiry.id} 后的新观察以重开研判')
    if missing:
        lines.extend(f'- {text}' for text in missing)
    else:
        lines.append('- 无')
    return '\n'.join(lines) + '\n'


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='色漂批源研判与隔离案卷')
    sub = parser.add_subparsers(dest='command', required=True)

    p_open = sub.add_parser('open', help='建立案卷并输出首版研判')
    p_open.add_argument('data_dir', type=Path)
    p_open.add_argument('--case', type=Path, required=True)

    p_reopen = sub.add_parser('reopen', help='以新资料重开研判')
    p_reopen.add_argument('case_dir', type=Path)
    p_reopen.add_argument('new_data_dir', type=Path)
    p_reopen.add_argument('--observation', default='')

    p_report = sub.add_parser('report', help='输出最新研判报告')
    p_report.add_argument('data_dir', type=Path)

    p_action = sub.add_parser('action', help='登记已执行的停用/通知/放行事实')
    p_action.add_argument('case_dir', type=Path)
    p_action.add_argument('action_type', choices=('hold', 'notify', 'release'))
    p_action.add_argument('job_id')
    p_action.add_argument('--detail', default='')

    args = parser.parse_args(argv)
    if args.command == 'open':
        case = open_case(args.case, args.data_dir)
        print(build_report(args.data_dir))
        print(f'案卷已写入 {case.directory}')
    elif args.command == 'reopen':
        case = reopen_case(args.case_dir, args.new_data_dir, args.observation)
        print(build_report(args.new_data_dir))
        print('已执行事实保留于事件日志：')
        for event in case.executed_actions():
            print(json.dumps(event, ensure_ascii=False))
    elif args.command == 'report':
        print(build_report(args.data_dir))
    elif args.command == 'action':
        from .casefile import CaseFile

        case = CaseFile(args.case_dir)
        event = case.record_action(args.action_type, args.job_id, args.detail)
        print(json.dumps(event, ensure_ascii=False))


if __name__ == '__main__':
    main()
