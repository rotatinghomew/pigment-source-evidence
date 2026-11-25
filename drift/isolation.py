"""隔离位排程与补证候选选择。

隔离原则
--------
* 分析的不确定性不等同全放或全扣：involved_all 必停，involved_some 待检，
  excluded 仅作“有条件放行”，须主管签认，任何结论都不得写成无条件出货。
* 已发件不占隔离位，进入通知待审范围。
* 一个在制订单不能跨物理隔离位拆放；多个订单可共享隔离位但受限量约束。
* 本班排不下的待检件保留在清单上并注明所缺资料（隔离位/工位窗口），
  不得静默丢弃。

补证原则
--------
候选补称、取留样、拆箱共享同一班组的人力与停线窗口；选择依据是
“能排除多少彼此冲突的可能解释”，而不是只测最大的一单。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from .analyze import (
    EXCLUDED,
    INVOLVED_ALL,
    INVOLVED_SOME,
    CaseAnalysis,
    VERDICT_ORDER,
)
from .model import Inquiry, Scale, Slot, TOL


@dataclass
class IsolationPlan:
    # 已发件：通知待审
    notify: list[str] = field(default_factory=list)
    # 在制件 -> 隔离位 id
    assigned: dict[str, str] = field(default_factory=dict)
    # 隔离位 id -> 其上的工件与已用量
    slot_usage: dict[str, list[str]] = field(default_factory=dict)
    # 本班无法安置的在制件及原因
    unplaced: dict[str, str] = field(default_factory=dict)
    # 有条件放行（excluded），需主管签认，不是无条件出货
    conditional_release: list[str] = field(default_factory=list)
    # 已被资料预占的隔离位
    pre_occupied: dict[str, str] = field(default_factory=dict)


def plan_isolation(analysis: CaseAnalysis, slots: list[Slot]) -> IsolationPlan:
    plan = IsolationPlan()
    for slot in slots:
        if slot.occupied_by:
            plan.pre_occupied[slot.id] = slot.occupied_by
            plan.slot_usage.setdefault(slot.id, [slot.occupied_by])

    # 已被资料预占隔离位的在制件视为已执行停用，不再参与重排。
    pre_occupied_jobs = set(plan.pre_occupied.values())
    floor_jobs = [
        verdict for verdict in analysis.verdicts.values()
        if verdict.state != 'sent' and verdict.job_id not in pre_occupied_jobs
    ]
    # 必停优先于待检；同级按重量从大到小，先安置最难放的件。
    floor_jobs.sort(key=lambda v: (VERDICT_ORDER[v.classification], -v.weight, v.job_id))

    remaining = {
        slot.id: slot.limit - sum(
            analysis.verdicts[j].weight
            for j in plan.slot_usage.get(slot.id, [])
            if j in analysis.verdicts
        )
        for slot in slots
    }

    def fits(slot: Slot, verdict) -> bool:
        # 隔离位窗口（start/end）是该位置可供占用的时段，与工件用料时间轴
        # 不同期；能否安置取决于工位适配与剩余限量，窗口仅作台账核查属性。
        return (
            verdict.position in slot.positions
            and remaining[slot.id] + TOL >= verdict.weight
        )

    for verdict in floor_jobs:
        if verdict.classification == EXCLUDED:
            plan.conditional_release.append(verdict.job_id)
            continue
        options = [
            slot for slot in slots
            if fits(slot, verdict)
        ]
        if not options:
            reason = _block_reason(verdict, slots, remaining)
            plan.unplaced[verdict.job_id] = reason
            continue
        # 选剩余容量最紧的适配位，给后续大件留余地（best-fit）。
        chosen = min(options, key=lambda s: (remaining[s.id], s.id))
        plan.assigned[verdict.job_id] = chosen.id
        plan.slot_usage.setdefault(chosen.id, []).append(verdict.job_id)
        remaining[chosen.id] -= verdict.weight

    plan.notify = sorted(
        verdict.job_id for verdict in analysis.verdicts.values()
        if verdict.state == 'sent'
        and verdict.classification in (INVOLVED_ALL, INVOLVED_SOME)
    )
    plan.conditional_release.sort()
    return plan


def _block_reason(verdict, slots, remaining) -> str:
    positional = [s for s in slots if verdict.position in s.positions]
    if not positional:
        return f'无适配工位 {verdict.position} 的隔离位'
    overloaded = [
        s.id for s in positional
        if remaining[s.id] + TOL < verdict.weight
    ]
    if len(overloaded) == len(positional):
        return '适配隔离位剩余容量不足，需追加隔离位或转出占用件'
    return '现有隔离位组合无法同时满足工位与限量约束，需追加资源'


@dataclass
class InquiryOption:
    inquiry: Inquiry
    value: float
    rationale: str


@dataclass
class InquirySchedule:
    # 推荐组合（按价值降序）
    selected: list[InquiryOption]
    # 全部互斥关系说明
    conflicts: list[str]
    # 每个候选的独立评估，供接手人改判
    options: dict[str, InquiryOption]
    # 每个可行班组组合的总价值，便于复盘
    alternative_sets: list[tuple[tuple[str, ...], float]]


def schedule_inquiries(
    analysis: CaseAnalysis,
    inquiries: list[Inquiry],
    scales: dict[str, Scale],
) -> InquirySchedule:
    options = {
        inquiry.id: _score_inquiry(inquiry, analysis, scales)
        for inquiry in inquiries
    }

    # 同班组且时间窗相交的候选互斥；跨班组可并行。
    conflicts: list[str] = []
    for left, right in combinations(inquiries, 2):
        if left.crew != right.crew:
            continue
        if left.start <= right.end + TOL and right.start <= left.end + TOL:
            conflicts.append(
                f'{left.id} 与 {right.id} 共享班组 {left.crew} 且窗口相交，只能二选一'
            )

    feasible_sets: list[tuple[tuple[str, ...], float]] = []
    ids = [inquiry.id for inquiry in inquiries]
    for size in range(len(ids), 0, -1):
        for combo in combinations(ids, size):
            if _internally_feasible(combo, inquiries):
                total = round(sum(options[i].value for i in combo), 6)
                feasible_sets.append((combo, total))
    feasible_sets.sort(key=lambda item: (-item[1], item[0]))

    best_ids = feasible_sets[0][0] if feasible_sets else ()
    selected = [options[i] for i in sorted(best_ids, key=lambda i: (-options[i].value, i))]
    return InquirySchedule(
        selected=selected,
        conflicts=conflicts,
        options=options,
        alternative_sets=feasible_sets,
    )


def _internally_feasible(combo: tuple[str, ...], inquiries: list[Inquiry]) -> bool:
    chosen = [q for q in inquiries if q.id in combo]
    for left, right in combinations(chosen, 2):
        if left.crew == right.crew and (
            left.start <= right.end + TOL and right.start <= left.end + TOL
        ):
            return False
    return True


def _score_inquiry(
    inquiry: Inquiry, analysis: CaseAnalysis, scales: dict[str, Scale]
) -> InquiryOption:
    """证据价值：以“能排除多少彼此冲突的解释”计分，不按订单大小给分。"""

    verdicts = analysis.verdicts
    if inquiry.kind == 'weigh':
        # 称量复核：若目标是被补称/补称凭证，决定补称场景取舍，影响该罐段
        # 全部工件的来源关系；一般称量复核则收紧该凭证的用量不确定性。
        target_scale = scales.get(inquiry.target)
        affected: set[str] = set()
        is_repair = bool(target_scale and target_scale.repair_of) or any(
            s.repair_of == inquiry.target for s in scales.values()
        )
        if target_scale is not None:
            for result in analysis.scenarios:
                if not result.feasible:
                    continue
                for jid, verdict in verdicts.items():
                    if inquiry.target in result.model.candidates.get(jid, ()):
                        affected.add(jid)
        n_scenarios = len([s for s in analysis.scenarios if s.feasible])
        value = (3.0 if is_repair else 2.0) + 0.5 * len(affected)
        rationale = (
            f'复核称量 {inquiry.target}'
            + ('（补称凭证，可在场景间取舍）' if is_repair else '')
            + f'，牵动 {len(affected)} 个工件的来源关系，共 {n_scenarios} 个可行场景'
        )
    elif inquiry.kind == 'retain':
        verdict = verdicts.get(inquiry.target)
        if verdict is None:
            value, rationale = 0.0, '目标工件不存在'
        else:
            uncertain = len(verdict.possible_batches - verdict.necessary_batches)
            if verdict.anomalous:
                value = 2.0 + 0.5 * uncertain
                rationale = f'为已判异常的 {inquiry.target} 增加一次独立测量，锁定相容批'
            elif verdict.classification == EXCLUDED:
                value = 0.5
                rationale = f'{inquiry.target} 已被现有资料排除，复测仅作放行背书'
            else:
                value = 2.5 + 0.5 * uncertain
                rationale = (
                    f'{inquiry.target} 未留样或状态未定，补留样可直接区分'
                    f' {sorted(verdict.possible_batches)} 中冲突的来源解释'
                )
    elif inquiry.kind == 'unpack':
        verdict = verdicts.get(inquiry.target)
        if verdict is None:
            value, rationale = 0.0, '目标工件不存在'
        elif verdict.classification == EXCLUDED:
            value = 0.5
            rationale = f'{inquiry.target} 已排除涉入，拆箱只验证签认放行，证据增量低'
        else:
            value = 1.5 + 0.5 * len(verdict.possible_batches)
            rationale = f'拆箱 {inquiry.target} 可核对实物与留样状态，缩小候选批范围'
    else:
        value, rationale = 0.0, '未知补证类型'
    return InquiryOption(inquiry, round(value, 6), rationale)
