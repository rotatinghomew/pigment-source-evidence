"""色漂研判核心规则测试（基于仓库自带合成资料）。"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from drift.analyze import (
    EXCLUDED,
    INVOLVED_ALL,
    INVOLVED_SOME,
    analyze_case,
)
from drift.casefile import CaseFile, open_case, reopen_case
from drift.isolation import plan_isolation, schedule_inquiries
from drift.model import (
    FeasibilityModel,
    build_scenarios,
    load_records,
    scale_can_feed,
)

DATA = Path(__file__).resolve().parents[1] / 'data'


class AnalysisTestBase(unittest.TestCase):
    def setUp(self):
        (
            self.batches, self.scales, self.jobs, self.retains,
            self.slots, self.inquiries, self.empties,
        ) = load_records(DATA)
        self.analysis = analyze_case(
            self.batches, self.scales, self.jobs, self.retains, self.empties
        )


class ScenarioTests(AnalysisTestBase):
    def test_both_repair_scenarios_are_kept_and_feasible(self):
        scenarios = build_scenarios(self.scales)
        self.assertEqual({s.name for s in scenarios}, {'baseline', 'S6>S2'})
        by_name = {s.name: s for s in scenarios}
        # 补称是替换件：baseline 只认原称 S2，不含补称 S6。
        self.assertNotIn('S6', by_name['baseline'].active_scales)
        self.assertIn('S2', by_name['baseline'].active_scales)
        repair = by_name['S6>S2']
        self.assertEqual(repair.superseded, frozenset({'S2'}))
        self.assertIn('S6', repair.active_scales)
        self.assertNotIn('S2', repair.active_scales)
        for result in self.analysis.scenarios:
            self.assertTrue(result.feasible)
            self.assertTrue(result.evidence_feasible)

    def test_repair_changes_whole_vessel_segment_not_single_order(self):
        # S6 替换 S2 影响 M1 罐段内 J1 与 J2 两个工件，而非只改一张订单。
        for result in self.analysis.scenarios:
            m1_jobs = [j for j in self.jobs if self.jobs[j].vessel == 'M1']
            for jid in m1_jobs:
                self.assertTrue(
                    any(
                        sid in result.model.candidates.get(jid, ())
                        for sid in ('S2', 'S6')
                    )
                )

    def test_empty_boundary_blocks_feed_across_clear(self):
        # M2 在 53 清空，清空之前的称量不可能供到清空之后（本资料该段无工件，
        # 这里直接验证时间谓词拒绝跨清空供料）。
        later_job = type(self.jobs['J5'])(
            'JX', 'M2', 54.0, 56.0, 10.0, 'red', 8.0, 10.0, 'P2', 'floor'
        )
        self.assertFalse(scale_can_feed(self.scales['S5'], later_job, self.empties))
        self.assertTrue(scale_can_feed(self.scales['S5'], self.jobs['J5'], self.empties))


class AnomalyAndExplanationTests(AnalysisTestBase):
    def test_anomaly_uses_product_spec_not_retain_band(self):
        # V1=[10.9,11.3] 与 J1 上限 10.3 无交集 -> 异常；
        # V2=[9.5,10.3] 落在 J4 范围 [8,10.3] -> 在标，不算异常。
        self.assertEqual(self.analysis.anomalous_jobs, ['J1', 'J3'])

    def test_common_culprit_and_ruled_out_batches(self):
        # B2 能同时解释 J1、J3；B4 无法解释 J1，被现有资料排除。
        self.assertEqual(self.analysis.minimal_groups, [frozenset({'B2'})])
        self.assertEqual(self.analysis.common_culprits, frozenset({'B2'}))
        self.assertEqual(self.analysis.ruled_out_batches, frozenset({'B4'}))

    def test_one_retain_is_one_piece_of_evidence(self):
        # 证据约束空间内，J1 必然同时含 B1 与 B2（单批无法落入 V1 误差带）。
        for result in self.analysis.scenarios:
            if not result.evidence_feasible:
                continue
            self.assertIn('B2', result.necessary_batches['J1'])
            self.assertIn('B1', result.necessary_batches['J1'])

    def test_necessity_under_quantity_cap(self):
        # B3 总量 30kg 上限 + V3 证据：J3 必然含 B3，且高漂批只能部分替代。
        for result in self.analysis.scenarios:
            self.assertIn('B3', result.necessary_batches['J3'])

    def test_job_three_way_classification(self):
        verdicts = self.analysis.verdicts
        self.assertEqual(verdicts['J1'].classification, INVOLVED_ALL)
        self.assertEqual(verdicts['J3'].classification, INVOLVED_ALL)
        self.assertEqual(verdicts['J2'].classification, INVOLVED_SOME)
        self.assertEqual(verdicts['J5'].classification, INVOLVED_SOME)

    def test_no_unconditional_release_for_any_job(self):
        # 本资料没有 excluded 件；即便有，隔离计划也只能给“有条件放行”。
        self.assertNotIn(
            EXCLUDED, {v.classification for v in self.analysis.verdicts.values()}
        )

    def test_missing_retain_is_not_treated_as_normal(self):
        # J2、J5 没有留样：未留样不是正常，仍按候选高漂批判为部分涉入待检。
        self.assertEqual(
            {jid for jid in ('J2', 'J5') if not self.analysis.verdicts[jid].retain_ids},
            {'J2', 'J5'},
        )
        for jid in ('J2', 'J5'):
            self.assertEqual(
                self.analysis.verdicts[jid].classification, INVOLVED_SOME
            )

    def test_in_spec_retain_does_not_grant_unconditional_release(self):
        # J4 的留样 V2 落在订单范围内，但它仍可能含高漂批，只能待检而非放行。
        verdict = self.analysis.verdicts['J4']
        self.assertFalse(verdict.anomalous)
        self.assertEqual(verdict.classification, INVOLVED_SOME)
        self.assertIn('B2', verdict.possible_batches)

    def test_evidence_range_respects_retain_bands(self):
        # 证据相容色值区间必须落在该工件全部留样误差带的并集包络内，
        # 且与物理包络采用可区分的两套口径。
        retain_bounds = {}
        for retain in self.retains:
            low, high = retain.interval
            if retain.job not in retain_bounds:
                retain_bounds[retain.job] = [low, high]
            else:
                retain_bounds[retain.job][0] = min(retain_bounds[retain.job][0], low)
                retain_bounds[retain.job][1] = max(retain_bounds[retain.job][1], high)
        for jid, (low, high) in retain_bounds.items():
            e_low, e_high = self.analysis.verdicts[jid].evidence_range
            self.assertGreaterEqual(e_low, low - 1e-6)
            self.assertLessEqual(e_high, high + 1e-6)


class IsolationTests(AnalysisTestBase):
    def setUp(self):
        super().setUp()
        self.plan = plan_isolation(self.analysis, self.slots)

    def test_sent_job_goes_to_notify_without_slot(self):
        self.assertIn('J1', self.plan.notify)
        self.assertNotIn('J1', self.plan.assigned)

    def test_preoccupied_slot_is_not_double_booked(self):
        # I1 已被 J2 预占，J2 不再参与重排，也不会出现在未安置清单。
        self.assertEqual(self.plan.pre_occupied, {'I1': 'J2'})
        self.assertNotIn('J2', self.plan.assigned)
        self.assertNotIn('J2', self.plan.unplaced)

    def test_job_is_never_split_across_slots(self):
        for jid, slot_id in self.plan.assigned.items():
            self.assertIsInstance(slot_id, str)
            self.assertIn(jid, self.plan.slot_usage[slot_id])

    def test_capacity_shortage_is_kept_visible_not_silently_released(self):
        # J5 无容量安置，必须保留在未安置清单而非给出货结论。
        self.assertIn('J5', self.plan.unplaced)
        self.assertNotIn('J5', self.plan.assigned)

    def test_slot_weight_limits_respected(self):
        for slot in self.slots:
            occupants = self.plan.slot_usage.get(slot.id, [])
            used = sum(
                self.analysis.verdicts[j].weight
                for j in occupants
                if j in self.analysis.verdicts
            )
            self.assertLessEqual(used, slot.limit + 1e-7)


class InquiryTests(AnalysisTestBase):
    def test_same_crew_overlap_is_mutex(self):
        schedule = schedule_inquiries(self.analysis, self.inquiries, self.scales)
        self.assertTrue(
            any('K1' in text and 'K2' in text for text in schedule.conflicts)
        )
        selected = {opt.inquiry.id for opt in schedule.selected}
        self.assertFalse({'K1', 'K2'} <= selected)

    def test_selection_ranks_by_explanations_not_order_size(self):
        schedule = schedule_inquiries(self.analysis, self.inquiries, self.scales)
        values = {iid: opt.value for iid, opt in schedule.options.items()}
        # 给未留样且未安置的 J5 补留样，价值高于对最大订单相关称量的复核。
        self.assertGreater(values['K2'], values['K1'])
        self.assertEqual(schedule.selected[0].inquiry.id, 'K2')

    def test_cross_crew_inquiries_run_in_parallel(self):
        schedule = schedule_inquiries(self.analysis, self.inquiries, self.scales)
        selected = {opt.inquiry.id for opt in schedule.selected}
        # Q2 的 K3 不与 Q1 候选争用人力，应可同时入选。
        self.assertIn('K3', selected)


class CaseFileTests(AnalysisTestBase):
    def test_open_and_reopen_keeps_executed_facts(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / 'case'
            case = open_case(case_dir, DATA)
            case.record_action('notify', 'J1', '已发件追发停用通知')
            case.record_action('hold', 'J3', '移入 I3')

            # 用同资料重开（模拟新观察到达后的重研判）。
            reopen_case(case_dir, DATA, observation='现场补回一次称重读数')
            reloaded = CaseFile(case_dir)
            actions = reloaded.executed_actions()
            self.assertEqual([a['job_id'] for a in actions], ['J1', 'J3'])
            # 事件序列：首版快照、2 条动作、观察、重开快照，共 5 条。
            self.assertEqual(len(reloaded.events()), 5)
            kinds = [e['kind'] for e in reloaded.events()]
            self.assertEqual(kinds.count('action'), 2)
            self.assertEqual(kinds.count('analysis'), 2)
            self.assertEqual(kinds.count('observation'), 1)
            # 已执行动作不会被新快照改写。
            self.assertTrue(
                all(e['kind'] != 'action' or e['payload']['job_id'] in ('J1', 'J3')
                    for e in reloaded.events())
            )

    def test_open_twice_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / 'case'
            open_case(case_dir, DATA)
            with self.assertRaises(FileExistsError):
                open_case(case_dir, DATA)

    def test_snapshot_contains_alternatives_and_missing_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = open_case(Path(tmp) / 'case', DATA)
            snapshot = case.latest_analysis()
            self.assertEqual(
                len(snapshot['analysis']['scenarios']), 2
            )
            self.assertIn('isolation_plan', snapshot)
            self.assertIn('inquiry_schedule', snapshot)
            self.assertTrue(snapshot['isolation_plan']['unplaced'])


class PhysicalConservationTests(AnalysisTestBase):
    def test_mass_cannot_be_invented(self):
        # 工件时间窗推到所有称量之后 -> 无时间相容来源 -> 场景不可行，
        # 模型不会用名义上从未投入的质量凑出该工件。
        broken_jobs = dict(self.jobs)
        j5 = broken_jobs['J5']
        broken_jobs['J5'] = type(j5)(
            j5.id, j5.vessel, 90.0, 95.0, j5.weight, j5.shade,
            j5.spec_min, j5.spec_max, j5.position, j5.state,
        )
        scenario = build_scenarios(self.scales)[0]
        model = FeasibilityModel.build(
            scenario, self.batches, self.scales, broken_jobs,
            self.retains, self.empties,
        )
        self.assertEqual(model.candidates['J5'], [])
        self.assertFalse(model.feasible(with_evidence=False))

    def test_batch_quantity_cap_binds(self):
        # B3 可贡献总量只有 30kg：在同一个联合分配里最大化 M2 三件的 B3 用量，
        # 结果不得超过 30kg，且因三件共 64kg，剩余部分必须来自高漂批。
        from scipy.optimize import linprog

        model = FeasibilityModel.build(
            build_scenarios(self.scales)[0],
            self.batches, self.scales, self.jobs, self.retains, self.empties,
        )
        self.assertTrue(model.feasible(with_evidence=True))
        vars_, index = model._variables()
        a_ub, b_ub, a_eq, b_eq = model._all_constraints(index, with_evidence=True)
        objective = [0.0] * len(vars_)
        for (jid, sid), col in index.items():
            if jid in ('J3', 'J4', 'J5') and self.scales[sid].batch == 'B3':
                objective[col] = -1.0
        result = linprog(
            objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
            bounds=[(0.0, None)] * len(index), method='highs',
        )
        max_b3 = -result.fun
        self.assertLessEqual(max_b3, self.batches['B3'].quantity + 1e-7)
        self.assertLess(max_b3, 64.0 - 1e-7)


if __name__ == '__main__':
    unittest.main()
