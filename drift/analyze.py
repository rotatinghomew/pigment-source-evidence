"""跨场景研判：异常判定、候选批相容性、共同解释与工件三分类。

输出刻意保留“所有能同时成立的解释”，不为每个工件单点挑一个最像来源。
分类口径：
* involved_all   —— 在每一个可行场景/解释中都涉入（必停、必通知）；
* involved_some  —— 仅在部分解释中涉入（待检，受本班隔离位约束）；
* excluded       —— 现有资料已排除其涉入异常，但不等于无条件放行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from .model import (
    TOL,
    Batch,
    FeasibilityModel,
    Retain,
    Scale,
    Scenario,
    build_scenarios,
)

INVOLVED_ALL = 'involved_all'
INVOLVED_SOME = 'involved_some'
EXCLUDED = 'excluded'

# 工件分类的展示次序，也用于隔离排程的优先级。
VERDICT_ORDER = {INVOLVED_ALL: 0, INVOLVED_SOME: 1, EXCLUDED: 2}


@dataclass
class ScenarioResult:
    scenario: Scenario
    model: FeasibilityModel
    feasible: bool
    evidence_feasible: bool
    # 工件 -> 可达到混合色值区间
    blends: dict[str, tuple[float, float]]
    # 留样 id -> 与场景是否相容
    retain_compatible: dict[str, bool]
    # 判定为异常的留样 id
    anomalous_retains: list[str]
    # 工件 -> 可能供到它的批
    possible_batches: dict[str, set[str]]
    # 工件 -> 必然供到它的批
    necessary_batches: dict[str, set[str]]


@dataclass
class JobVerdict:
    job_id: str
    state: str
    position: str
    weight: float
    spec: tuple[float, float]
    classification: str
    # 跨可行场景合并后的可能/必然批源
    possible_batches: set[str] = field(default_factory=set)
    necessary_batches: set[str] = field(default_factory=set)
    # 与该工件有关的留样（含异常与在标）
    retain_ids: list[str] = field(default_factory=list)
    anomalous: bool = False
    reasons: list[str] = field(default_factory=list)
    # 物理可行分配下的标称包络区间（批标称极值）
    achievable_range: tuple[float, float] | None = None
    # 全部留样证据相容下的混合读数区间（批中心值口径）
    evidence_range: tuple[float, float] | None = None


@dataclass
class CaseAnalysis:
    batches: dict[str, Batch]
    scales: dict[str, Scale]
    jobs: dict
    retains: list[Retain]
    scenarios: list[ScenarioResult]
    verdicts: dict[str, JobVerdict]
    # 能够单独/共同解释全部当前异常的高漂批集合（最小充分组）
    minimal_groups: list[frozenset[str]]
    # 跨所有最小充分组都出现的批（共同根因候选）
    common_culprits: frozenset[str]
    # 现有资料排除涉入的高漂批
    ruled_out_batches: frozenset[str]
    # 异常工件 id（在至少一个可行场景中留样超订单范围）
    anomalous_jobs: list[str]
    # 每个异常工件各自相容的高漂批
    anomaly_compatible_batches: dict[str, set[str]]


def _retain_is_anomaly(retain: Retain, job) -> bool:
    """色漂判据按产品订好的范围：留样误差带与订单范围无交集即判异常。"""

    low, high = retain.interval
    return high < job.spec_min - TOL or low > job.spec_max + TOL


def analyze_case(batches, scales, jobs, retains, empties) -> CaseAnalysis:
    scenarios = build_scenarios(scales)
    retains_by_job: dict[str, list[Retain]] = {}
    for retain in retains:
        retains_by_job.setdefault(retain.job, []).append(retain)

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        model = FeasibilityModel.build(
            scenario, batches, scales, jobs, retains, empties
        )
        # 物理可行性：时间/清空/用量约束。
        feasible = model.feasible(with_evidence=False)
        # 证据相容性：全部留样误差带能否同时满足。
        evidence_feasible = model.feasible(with_evidence=True) if feasible else False
        blends = {
            job_id: model.achievable_blend(job_id, with_evidence=False)
            for job_id in jobs
        } if feasible else {}
        compatible: dict[str, bool] = {}
        anomalous: list[str] = []
        if feasible:
            for retain in retains:
                ok = model.retain_compatible(retain.job, retain)
                compatible[retain.id] = ok
                if ok and _retain_is_anomaly(retain, jobs[retain.job]):
                    anomalous.append(retain.id)
        possible_batches: dict[str, set[str]] = {}
        necessary_batches: dict[str, set[str]] = {}
        if evidence_feasible:
            # 批源关系只在“与全部留样相容”的分配空间内判定，
            # 同一份留样经同一组约束生效，不拆成两个来源的独立证据。
            for job_id in jobs:
                possible_batches[job_id] = {
                    bid for bid in batches
                    if model.batch_possible(job_id, bid, with_evidence=True)
                }
                necessary_batches[job_id] = {
                    bid for bid in batches
                    if model.batch_necessary(job_id, bid, with_evidence=True)
                }
        results.append(ScenarioResult(
            scenario=scenario,
            model=model,
            feasible=feasible,
            evidence_feasible=evidence_feasible,
            blends=blends,
            retain_compatible=compatible,
            anomalous_retains=anomalous,
            possible_batches=possible_batches,
            necessary_batches=necessary_batches,
        ))

    feasible_results = [r for r in results if r.feasible]
    if not feasible_results:
        raise ValueError('没有任何补称场景同时满足时间与用量约束，资料互相冲突')

    # 高漂批：标称色值整体高于任何同色号订单上限的供应批；
    # 色号相关本身不推因果，只用于限定候选范围，因果仍由留样与约束相容性给出。
    shade_spec_max: dict[str, float] = {}
    for job in jobs.values():
        shade_spec_max[job.shade] = max(
            shade_spec_max.get(job.shade, float('-inf')), job.spec_max
        )
    high_batches = {
        bid for bid, batch in batches.items()
        if batch.shade in shade_spec_max
        and batch.value_min > shade_spec_max[batch.shade] + TOL
    }

    # 跨可行场景合并：某批“可能”指在任一可行场景中可能；“必然”指在全部可行场景中必然。
    merged_possible: dict[str, set[str]] = {jid: set() for jid in jobs}
    merged_necessary: dict[str, set[str]] = {jid: set(batches) for jid in jobs}
    merged_range: dict[str, list[float]] = {}
    evidence_range: dict[str, list[float]] = {}
    evidence_results = [r for r in feasible_results if r.possible_batches]
    if not evidence_results:
        raise ValueError('物理可行的场景均无法同时满足全部留样误差带，资料互相冲突')
    for result in evidence_results:
        for jid in jobs:
            merged_possible[jid] |= result.possible_batches[jid]
            merged_necessary[jid] &= result.necessary_batches[jid]
            low, high = result.blends[jid]
            if jid not in merged_range:
                merged_range[jid] = [low, high]
            else:
                merged_range[jid][0] = min(merged_range[jid][0], low)
                merged_range[jid][1] = max(merged_range[jid][1], high)
            e_low, e_high = result.model.achievable_blend(jid, with_evidence=True)
            if jid not in evidence_range:
                evidence_range[jid] = [e_low, e_high]
            else:
                evidence_range[jid][0] = min(evidence_range[jid][0], e_low)
                evidence_range[jid][1] = max(evidence_range[jid][1], e_high)

    # 异常留样 -> 工件；同一份留样在所有解释中只登记一次。
    anomaly_jobs: dict[str, list[str]] = {}
    for result in feasible_results:
        for rid in result.anomalous_retains:
            retain = next(r for r in retains if r.id == rid)
            anomaly_jobs.setdefault(retain.job, [])
            if rid not in anomaly_jobs[retain.job]:
                anomaly_jobs[retain.job].append(rid)

    # 每个异常工件相容的高漂批：必须在至少一个证据可行场景中可能供到该工件。
    anomaly_compatible: dict[str, set[str]] = {}
    for jid, rids in anomaly_jobs.items():
        compatible_batches: set[str] = set()
        for result in evidence_results:
            if all(result.retain_compatible[rid] for rid in rids):
                compatible_batches |= (
                    result.possible_batches.get(jid, set()) & high_batches
                )
        anomaly_compatible[jid] = compatible_batches

    # 共同解释：求能覆盖全部异常工件的最小高漂批集合（按集合包含极小化）。
    coverage = list(anomaly_compatible.values())
    minimal_groups = _minimal_covers(high_batches, coverage)

    common_culprits = frozenset(
        bid for bid in frozenset.union(*minimal_groups)
        if all(bid in group for group in minimal_groups)
    ) if minimal_groups else frozenset()

    ruled_out = frozenset(
        high_batches - frozenset.union(*minimal_groups)
    ) if minimal_groups else frozenset(high_batches)

    verdicts: dict[str, JobVerdict] = {}
    for jid, job in jobs.items():
        job_retains = [r.id for r in retains if r.job == jid]
        is_anomalous = jid in anomaly_jobs
        possible_high = merged_possible[jid] & high_batches
        necessary_high = merged_necessary[jid] & high_batches

        reasons: list[str] = []
        if is_anomalous:
            reasons.append(
                '留样 ' + '、'.join(anomaly_jobs[jid])
                + ' 的误差带与订单范围无交集'
            )
        if necessary_high:
            reasons.append('在全部可行解释中都必然含高漂批 ' + '、'.join(sorted(necessary_high)))
        elif possible_high:
            reasons.append('在部分可行解释中可能含高漂批 ' + '、'.join(sorted(possible_high)))
        else:
            reasons.append('现有资料排除其含任何高漂批')

        if is_anomalous or necessary_high:
            classification = INVOLVED_ALL
        elif possible_high:
            classification = INVOLVED_SOME
        else:
            classification = EXCLUDED

        verdicts[jid] = JobVerdict(
            job_id=jid,
            state=job.state,
            position=job.position,
            weight=job.weight,
            spec=(job.spec_min, job.spec_max),
            classification=classification,
            possible_batches=set(merged_possible[jid]),
            necessary_batches=set(merged_necessary[jid]),
            retain_ids=job_retains,
            anomalous=is_anomalous,
            reasons=reasons,
            achievable_range=(merged_range[jid][0], merged_range[jid][1]),
            evidence_range=(evidence_range[jid][0], evidence_range[jid][1]),
        )

    return CaseAnalysis(
        batches=batches,
        scales=scales,
        jobs=jobs,
        retains=retains,
        scenarios=results,
        verdicts=verdicts,
        minimal_groups=minimal_groups,
        common_culprits=common_culprits,
        ruled_out_batches=ruled_out,
        anomalous_jobs=sorted(anomaly_jobs),
        anomaly_compatible_batches=anomaly_compatible,
    )


def _minimal_covers(universe: set[str], coverage: list[set[str]]) -> list[frozenset[str]]:
    """求覆盖 coverage 中每一组的极小集合族（集合包含意义下不可再删）。"""

    coverage = [set(group) for group in coverage if group]
    if not coverage:
        return []
    ordered = sorted(universe)
    minimal: list[frozenset[str]] = []
    for size in range(1, len(ordered) + 1):
        for combo in combinations(ordered, size):
            chosen = frozenset(combo)
            if any(chosen >= existing for existing in minimal):
                continue
            if all(chosen & group for group in coverage):
                minimal.append(chosen)
    return minimal
