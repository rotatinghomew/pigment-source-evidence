"""数据模型与约束可行性。

物理事实的建模原则
--------------------
* 罐以最后一次明确清空为界：清空之后的工件不可能用到清空前的装料。
* 称量窗口与用料窗口相交（含端点）才存在“供料可能”。
* 补称（repair_of）替换的是整张称量凭证，影响该罐该时间界内全部工件，
  不只影响某一张订单；因此每一种补称取舍构成一个独立场景，各场景分别求可行性。
* 每份留样是一次测量，误差带是读数区间，不是供料标签，也不能扩展订单标准；
  同一份留样在任意解释里只能使用一次，不拆成两个来源的独立证据。
* 混料不能攒出名义上从未投入的质量：每个工件的来源只能是与其时间相容的称量，
  且每罐段内总用量受各称量上限、供应批可贡献总量约束。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from scipy.optimize import linprog

# 浮点判定容差：仪表刻度与千克量均为小数量级，统一容差足够稳健。
TOL = 1e-7


def _num(value: str) -> float:
    return float(value) if value not in ('', None) else float('nan')


@dataclass(frozen=True)
class Batch:
    id: str
    shade: str
    value_min: float
    value_max: float
    quantity: float


@dataclass(frozen=True)
class Scale:
    id: str
    batch: str
    vessel: str
    start: float
    end: float
    visible: float
    weight_min: float
    weight_max: float
    repair_of: str = ''


@dataclass(frozen=True)
class Job:
    id: str
    vessel: str
    start: float
    end: float
    weight: float
    shade: str
    spec_min: float
    spec_max: float
    position: str
    state: str


@dataclass(frozen=True)
class Retain:
    id: str
    job: str
    measure_at: float
    visible: float
    value: float
    error: float

    @property
    def interval(self) -> tuple[float, float]:
        return self.value - self.error, self.value + self.error


@dataclass(frozen=True)
class Slot:
    id: str
    positions: frozenset[str]
    limit: float
    start: float
    end: float
    occupied_by: str = ''


@dataclass(frozen=True)
class Inquiry:
    id: str
    kind: str
    target: str
    start: float
    end: float
    crew: str


@dataclass
class Scenario:
    """一种补称取舍下的完整事实版本。"""

    name: str
    active_scales: frozenset[str]
    superseded: frozenset[str]
    note: str = ''

    def weight_bounds(self, scales: dict[str, Scale]) -> dict[str, tuple[float, float]]:
        return {
            sid: (s.weight_min, s.weight_max)
            for sid, s in scales.items()
            if sid in self.active_scales
        }


def load_records(directory: str | Path):
    """从交换资料目录读入强类型记录。结构与引用校验仍由 check_sources 负责。"""

    directory = Path(directory)

    def read(name: str) -> list[dict[str, str]]:
        with (directory / f'{name}.csv').open(encoding='utf-8', newline='') as fh:
            return list(csv.DictReader(fh))

    batches = {
        r['id']: Batch(
            r['id'], r['shade'], _num(r['value_min']), _num(r['value_max']),
            _num(r['quantity']),
        )
        for r in read('source_batches')
    }
    scales = {
        r['id']: Scale(
            r['id'], r['batch'], r['vessel'], _num(r['from']), _num(r['to']),
            _num(r['visible']), _num(r['weight_min']), _num(r['weight_max']),
            r['repair_of'],
        )
        for r in read('scale')
    }
    jobs = {
        r['id']: Job(
            r['id'], r['vessel'], _num(r['from']), _num(r['to']), _num(r['weight']),
            r['shade'], _num(r['min']), _num(r['max']), r['position'], r['state'],
        )
        for r in read('links')
    }
    retains = [
        Retain(
            r['id'], r['job'], _num(r['measure_at']), _num(r['visible']),
            _num(r['value']), _num(r['error']),
        )
        for r in read('retains')
    ]
    slots = [
        Slot(
            r['id'], frozenset(p.strip() for p in r['position'].split(',') if p.strip()),
            _num(r['limit']), _num(r['start']), _num(r['end']), r['occupied_by'],
        )
        for r in read('isolation')
    ]
    inquiries = [
        Inquiry(
            r['id'], r['kind'], r['target'], _num(r['start']), _num(r['end']), r['crew'],
        )
        for r in read('inquiries')
    ]

    empties: dict[str, list[float]] = {}
    for r in read('vessels'):
        if r['event'] == 'empty':
            empties.setdefault(r['vessel'], []).append(_num(r['at']))
    for points in empties.values():
        points.sort()

    return batches, scales, jobs, retains, slots, inquiries, empties


def segment_boundary(vessel: str, at: float, empties: dict[str, list[float]]) -> float:
    """返回 at 所在罐段的起点（最后一次不晚于 at 的清空；没有则为负无穷）。"""

    boundary = float('-inf')
    for point in empties.get(vessel, ()):  # 已排序
        if point <= at + TOL:
            boundary = point
        else:
            break
    return boundary


def scale_can_feed(scale: Scale, job: Job, empties: dict[str, list[float]]) -> bool:
    """时间与清空约束下，某张称量是否可能供到某个工件。

    称量窗口 [from, to] 表示该批料在窗口内某时刻完成入罐，因此只要入罐
    可能不晚于用料结束（from <= job.to）即存在供料可能，不要求两窗相交；
    可见时间 visible 只影响资料何时被看到，不改变物理时间关系。
    """

    if scale.vessel != job.vessel:
        return False
    if scale.start > job.end + TOL:
        return False
    # 跨越清空的称量不能供到清空之后；以工件所在罐段的清空界为准。
    boundary = segment_boundary(job.vessel, job.start, empties)
    if scale.end <= boundary + TOL:
        return False
    return True


def build_scenarios(scales: dict[str, Scale]) -> list[Scenario]:
    """按补称链生成互斥场景。

    每条 repair_of 链给出“维持原称”与“采纳补称”两种版本；
    不同链相互独立，取笛卡尔积。补称替换的是该罐段内的全部供料关系。
    """

    # 只保留以“被替换凭证 -> 替换凭证”为方向的链。
    replacements = {s.repair_of: s.id for s in scales.values() if s.repair_of}
    if not replacements:
        return [Scenario('baseline', frozenset(scales), frozenset())]

    # 补称凭证是替换件：baseline 只认原称，补称在“采纳补称”场景才启用。
    repair_ids = frozenset(replacements.values())
    baseline_active = frozenset(scales) - repair_ids
    scenarios: list[Scenario] = [
        Scenario('baseline', baseline_active, frozenset())
    ]
    for old_id, repair_id in replacements.items():
        grown: list[Scenario] = []
        for sc in scenarios:
            # 维持原称：当前场景不变。
            grown.append(sc)
            # 采纳补称：剔除原称、启用补称，影响该罐段内全部供料关系。
            active = frozenset((sc.active_scales - {old_id}) | {repair_id})
            superseded = sc.superseded | {old_id}
            name = f'{sc.name}+{repair_id}>{old_id}'.replace('baseline+', '')
            note = f'采纳补称 {repair_id} 替换 {old_id}'
            grown.append(Scenario(name, active, superseded, note))
        scenarios = grown
    return scenarios


@dataclass
class FeasibilityModel:
    """单一场景下，按罐段建立的质量分配线性可行性模型。"""

    scenario: Scenario
    batches: dict[str, Batch]
    scales: dict[str, Scale]
    jobs: dict[str, Job]
    retains: list[Retain]
    empties: dict[str, list[float]]
    # 罐段（vessel, 清空界）-> 该段启用的称量
    segment_scales: dict[tuple[str, float], list[str]] = field(default_factory=dict)
    # 工件 -> 候选称量（时间相容且场景启用）
    candidates: dict[str, list[str]] = field(default_factory=dict)
    _feasible: bool | None = None

    @classmethod
    def build(
        cls,
        scenario: Scenario,
        batches: dict[str, Batch],
        scales: dict[str, Scale],
        jobs: dict[str, Job],
        retains: list[Retain],
        empties: dict[str, list[float]],
    ) -> 'FeasibilityModel':
        model = cls(scenario, batches, scales, jobs, retains, empties)
        segments: dict[tuple[str, float], set[str]] = {}
        for job_id, job in jobs.items():
            boundary = segment_boundary(job.vessel, job.start, empties)
            key = (job.vessel, boundary)
            eligible = [
                sid
                for sid, scale in scales.items()
                if sid in scenario.active_scales and scale_can_feed(scale, job, empties)
            ]
            if not eligible:
                # 无任何时间相容来源：直接标记，求解时必然不可行。
                model.candidates[job_id] = []
                continue
            model.candidates[job_id] = sorted(eligible)
            segments.setdefault(key, set()).update(eligible)
        model.segment_scales = {
            key: sorted(ids) for key, ids in sorted(segments.items())
        }
        return model

    # ---- 约束矩阵 -----------------------------------------------------

    def _variables(self) -> tuple[list[tuple[str, str]], dict[tuple[str, str], int]]:
        vars_: list[tuple[str, str]] = []
        index: dict[tuple[str, str], int] = {}
        for job_id, eligible in self.candidates.items():
            for sid in eligible:
                index[(job_id, sid)] = len(vars_)
                vars_.append((job_id, sid))
        return vars_, index

    def _base_constraints(
        self, index: dict[tuple[str, str], int]
    ) -> tuple[list[list[float]], list[float], list[float]]:
        n = len(index)
        a_ub: list[list[float]] = []
        b_ub: list[float] = []
        a_eq: list[list[float]] = []
        b_eq: list[float] = []

        # 1) 每个工件按其总重量完整分配（不能凭空产生质量）。
        for job_id, job in self.jobs.items():
            row = [0.0] * n
            for sid in self.candidates.get(job_id, ()):
                row[index[(job_id, sid)]] = 1.0
            a_eq.append(row)
            b_eq.append(job.weight)

        # 2) 罐段内每张称量的总流出不超过其称带上限。
        for eligible in self.segment_scales.values():
            used: dict[str, list[tuple[str, str]]] = {}
            for (job_id, sid), col in index.items():
                if sid in eligible:
                    used.setdefault(sid, []).append((job_id, sid))
            for sid, cells in used.items():
                row = [0.0] * n
                for cell in cells:
                    row[index[cell]] = 1.0
                a_ub.append(row)
                b_ub.append(self.scales[sid].weight_max)

        # 3) 供应批可贡献总量（跨罐段全局上限）。
        by_batch: dict[str, list[int]] = {}
        for (job_id, sid), col in index.items():
            by_batch.setdefault(self.scales[sid].batch, []).append(col)
        for batch_id, cols in by_batch.items():
            row = [0.0] * n
            for col in cols:
                row[col] = 1.0
            a_ub.append(row)
            b_ub.append(self.batches[batch_id].quantity)
        return a_ub, b_ub, a_eq, b_eq

    def _retain_constraints(
        self, index: dict[tuple[str, str], int]
    ) -> tuple[list[list[float]], list[float]]:
        """把每份留样的误差带转成线性不等式（同一份留样只生成一组约束）。

        留样 value±error 是仪表读数，与按批标称中心值计算的混合读数比较：
        weight·(value-error) <= Σ center·x <= weight·(value+error)。
        批自身的 value_min/value_max 不确定性另由 achievable_blend 的
        包络区间表达，不在此重复收紧，避免把区间误差算两次。
        """
        a_ub: list[list[float]] = []
        b_ub: list[float] = []
        retains_by_job: dict[str, list[Retain]] = {}
        for retain in self.retains:
            retains_by_job.setdefault(retain.job, []).append(retain)
        n = len(index)
        for job_id, job_retains in retains_by_job.items():
            if job_id not in self.jobs:
                continue
            weight = self.jobs[job_id].weight
            for retain in job_retains:
                low, high = retain.interval
                # 下界：Σ center·x >= weight·low
                lower = [0.0] * n
                for sid in self.candidates.get(job_id, ()):
                    batch = self.batches[self.scales[sid].batch]
                    center = (batch.value_min + batch.value_max) / 2.0
                    lower[index[(job_id, sid)]] = -center
                a_ub.append(lower)
                b_ub.append(-weight * low)
                # 上界：Σ center·x <= weight·high
                upper = [0.0] * n
                for sid in self.candidates.get(job_id, ()):
                    batch = self.batches[self.scales[sid].batch]
                    center = (batch.value_min + batch.value_max) / 2.0
                    upper[index[(job_id, sid)]] = center
                a_ub.append(upper)
                b_ub.append(weight * high)
        return a_ub, b_ub

    def _all_constraints(
        self, index, with_evidence: bool
    ) -> tuple[list[list[float]], list[float], list[list[float]], list[float]]:
        a_ub, b_ub, a_eq, b_eq = self._base_constraints(index)
        if with_evidence:
            extra_a, extra_b = self._retain_constraints(index)
            a_ub = a_ub + extra_a
            b_ub = b_ub + extra_b
        return a_ub, b_ub, a_eq, b_eq

    def _solve(
        self,
        index: dict[tuple[str, str], int],
        a_ub: list[list[float]],
        b_ub: list[float],
        a_eq: list[list[float]],
        b_eq: list[float],
        objective: list[float] | None = None,
    ) -> bool:
        n = len(index)
        result = linprog(
            objective if objective is not None else [0.0] * n,
            A_ub=a_ub or None,
            b_ub=b_ub or None,
            A_eq=a_eq or None,
            b_eq=b_eq or None,
            bounds=[(0.0, None)] * n,
            method='highs',
        )
        return bool(result.success)

    def feasible(self, with_evidence: bool = False) -> bool:
        """约束是否整体可满足。

        with_evidence=False：仅时间/清空/用量约束；
        with_evidence=True：再加全部留样误差带，即“与已观测证据相容的解释”。
        """

        if not with_evidence and self._feasible is not None:
            return self._feasible
        if any(not eligible for eligible in self.candidates.values()):
            if not with_evidence:
                self._feasible = False
            return False
        _, index = self._variables()
        a_ub, b_ub, a_eq, b_eq = self._all_constraints(index, with_evidence)
        ok = self._solve(index, a_ub, b_ub, a_eq, b_eq)
        if not with_evidence:
            self._feasible = ok
        return ok

    def link_possible(self, job_id: str, scale_id: str, with_evidence: bool = True) -> bool:
        """某工件是否存在一种可行分配使其用到该称量（用量大于零）。"""

        if not self.feasible(with_evidence) or scale_id not in self.candidates.get(job_id, ()):
            return False
        return self._link_mass(job_id, scale_id, maximize=True, with_evidence=with_evidence) > TOL

    def link_necessary(self, job_id: str, scale_id: str, with_evidence: bool = True) -> bool:
        """该工件的任意可行分配是否都必须用到该称量。"""

        if not self.feasible(with_evidence) or scale_id not in self.candidates.get(job_id, ()):
            return False
        return self._link_mass(job_id, scale_id, maximize=False, with_evidence=with_evidence) > TOL

    def _link_mass(
        self, job_id: str, scale_id: str, maximize: bool, with_evidence: bool
    ) -> float:
        vars_, index = self._variables()
        a_ub, b_ub, a_eq, b_eq = self._all_constraints(index, with_evidence)
        col = index[(job_id, scale_id)]
        objective = [0.0] * len(vars_)
        objective[col] = -1.0 if maximize else 1.0
        result = linprog(
            objective,
            A_ub=a_ub or None, b_ub=b_ub or None,
            A_eq=a_eq or None, b_eq=b_eq or None,
            bounds=[(0.0, None)] * len(vars_),
            method='highs',
        )
        if not result.success:
            return 0.0
        return abs(result.x[col]) if maximize else max(0.0, result.x[col])

    def batch_possible(self, job_id: str, batch_id: str, with_evidence: bool = True) -> bool:
        return any(
            self.scales[sid].batch == batch_id
            and self.link_possible(job_id, sid, with_evidence)
            for sid in self.candidates.get(job_id, ())
        )

    def batch_necessary(self, job_id: str, batch_id: str, with_evidence: bool = True) -> bool:
        """该工件的任意相容分配是否都必须从该批取料（可能跨多张称量）。"""

        if not self.feasible(with_evidence):
            return False
        vars_, index = self._variables()
        a_ub, b_ub, a_eq, b_eq = self._all_constraints(index, with_evidence)
        cols = [
            col for (jid, sid), col in index.items()
            if jid == job_id and self.scales[sid].batch == batch_id
        ]
        if not cols:
            return False
        # 最小化该批总取料量：若最小值仍大于零，则该批不可避免。
        objective = [0.0] * len(vars_)
        for col in cols:
            objective[col] = 1.0
        result = linprog(
            objective,
            A_ub=a_ub or None, b_ub=b_ub or None,
            A_eq=a_eq or None, b_eq=b_eq or None,
            bounds=[(0.0, None)] * len(vars_),
            method='highs',
        )
        if not result.success:
            return False
        return sum(result.x[c] for c in cols) > TOL

    def achievable_blend(
        self, job_id: str, with_evidence: bool = False
    ) -> tuple[float, float]:
        """在可行质量分配下，该工件混合色值可达到的闭区间。

        混合值 = Σ(批色值 × 分配质量) / 工件总质量；
        线性规划分别求最小/最大加权色值。with_evidence=True 时限缩到
        全部留样误差带共同允许的分配空间。
        """

        if not self.feasible(with_evidence):
            return float('nan'), float('nan')
        job = self.jobs[job_id]
        vars_, index = self._variables()
        a_ub, b_ub, a_eq, b_eq = self._all_constraints(index, with_evidence)
        values: list[float] = []
        for use_max in (False, True):
            # 最小化/最大化该工件的 Σ 批色值·分配质量；
            # 等式 Σ质量=weight 保证结果除以 weight 后即混合色值。
            # 证据空间与留样约束采用同一套批中心值口径；物理包络（无证据）
            # 才用批标称极值，避免把批区间不确定性与留样误差重复计算。
            objective = [0.0] * len(vars_)
            for (jid, sid), col in index.items():
                if jid != job_id:
                    continue
                batch = self.batches[self.scales[sid].batch]
                if with_evidence:
                    coefficient = (batch.value_min + batch.value_max) / 2.0
                else:
                    coefficient = batch.value_max if use_max else batch.value_min
                objective[col] = coefficient
            result = linprog(
                # 求最大值时取负号做最小化。
                [c * (-1.0 if use_max else 1.0) for c in objective],
                A_ub=a_ub or None, b_ub=b_ub or None,
                A_eq=a_eq or None, b_eq=b_eq or None,
                bounds=[(0.0, None)] * len(vars_),
                method='highs',
            )
            if not result.success:
                return float('nan'), float('nan')
            raw = result.fun * (-1.0 if use_max else 1.0)
            values.append(raw / job.weight)
        return values[0], values[1]

    def retain_compatible(self, job_id: str, retain: Retain) -> bool:
        """留样误差带与可达到的混合色值区间是否存在交集。

        留样只作一次测量使用：这里检验的是“整罐分配能否落到该读数区间”，
        而不是把区间拆给多个来源分别作证。
        """

        low, high = self.achievable_blend(job_id)
        if low != low:  # NaN：场景不可行
            return False
        r_low, r_high = retain.interval
        return high + TOL >= r_low and low - TOL <= r_high
