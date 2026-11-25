"""隔离案卷：只追加的事件记录与重开研判。

事实与判断分开存储：
* action 事件（停用/通知）一旦登记即为已执行事实，重开研判不会删除或改写；
* analysis 事件是某一时点基于当时资料的判断快照，仅被后续快照取代；
* observation 事件登记现场新观察（新留样、补称核实结果等），重开时以新资料
  重新求解，并只更正尚未执行的停用/通知范围。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .analyze import analyze_case
from .isolation import IsolationPlan, plan_isolation, schedule_inquiries
from .model import load_records

EVENT_ANALYSIS = 'analysis'
EVENT_ACTION = 'action'
EVENT_OBSERVATION = 'observation'


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _analysis_to_dict(analysis) -> dict:
    return {
        'scenarios': [
            {
                'name': r.scenario.name,
                'feasible': r.feasible,
                'evidence_feasible': r.evidence_feasible,
                'superseded': sorted(r.scenario.superseded),
                'note': r.scenario.note,
                'anomalous_retains': r.anomalous_retains,
            }
            for r in analysis.scenarios
        ],
        'verdicts': {
            jid: {
                'classification': v.classification,
                'state': v.state,
                'position': v.position,
                'weight': v.weight,
                'spec': list(v.spec),
                'possible_batches': sorted(v.possible_batches),
                'necessary_batches': sorted(v.necessary_batches),
                'retain_ids': v.retain_ids,
                'anomalous': v.anomalous,
                'achievable_range': list(v.achievable_range)
                if v.achievable_range else None,
                'evidence_range': list(v.evidence_range)
                if v.evidence_range else None,
                'reasons': v.reasons,
            }
            for jid, v in analysis.verdicts.items()
        },
        'minimal_groups': [sorted(g) for g in analysis.minimal_groups],
        'common_culprits': sorted(analysis.common_culprits),
        'ruled_out_batches': sorted(analysis.ruled_out_batches),
        'anomalous_jobs': analysis.anomalous_jobs,
        'anomaly_compatible_batches': {
            jid: sorted(bids)
            for jid, bids in analysis.anomaly_compatible_batches.items()
        },
    }


def _plan_to_dict(plan: IsolationPlan) -> dict:
    return {
        'notify': plan.notify,
        'assigned': plan.assigned,
        'slot_usage': plan.slot_usage,
        'unplaced': plan.unplaced,
        'conditional_release': plan.conditional_release,
        'pre_occupied': plan.pre_occupied,
    }


@dataclass
class CaseFile:
    directory: Path

    @property
    def events_path(self) -> Path:
        return self.directory / 'events.jsonl'

    @property
    def meta_path(self) -> Path:
        return self.directory / 'case.json'

    def _append(self, kind: str, payload: dict) -> dict:
        self.directory.mkdir(parents=True, exist_ok=True)
        event = {'at': _utc_now(), 'kind': kind, 'payload': payload}
        with self.events_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + '\n')
        return event

    def events(self) -> list[dict]:
        if not self.events_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.events_path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]

    def latest_analysis(self) -> dict | None:
        for event in reversed(self.events()):
            if event['kind'] == EVENT_ANALYSIS:
                return event['payload']
        return None

    def executed_actions(self) -> list[dict]:
        return [
            event['payload']
            for event in self.events()
            if event['kind'] == EVENT_ACTION
        ]

    def record_action(self, action_type: str, job_id: str, detail: str = '') -> dict:
        """登记一条已执行事实（停用 hold / 通知 notify / 放行签认 release）。"""

        if action_type not in ('hold', 'notify', 'release'):
            raise ValueError('action_type 必须是 hold、notify 或 release')
        return self._append(EVENT_ACTION, {
            'action': action_type,
            'job_id': job_id,
            'detail': detail,
        })

    def record_observation(self, description: str, data: dict | None = None) -> dict:
        return self._append(EVENT_OBSERVATION, {
            'description': description,
            'data': data or {},
        })


def open_case(case_dir: str | Path, data_dir: str | Path) -> CaseFile:
    """建立案卷并写入首版研判快照。"""

    case = CaseFile(Path(case_dir))
    if case.events_path.exists():
        raise FileExistsError(f'案卷已存在：{case_dir}，重开请使用 reopen_case')
    case.directory.mkdir(parents=True, exist_ok=True)
    case.meta_path.write_text(json.dumps({
        'created_at': _utc_now(),
        'data_dir': str(data_dir),
        'revision': 0,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    _snapshot(case, data_dir)
    return case


def reopen_case(
    case_dir: str | Path,
    new_data_dir: str | Path,
    observation: str = '',
    observation_data: dict | None = None,
) -> CaseFile:
    """以新资料重开研判：追加观察与新快照，已执行的停用/通知事实原样保留。"""

    case = CaseFile(Path(case_dir))
    if not case.events_path.exists():
        raise FileNotFoundError(f'案卷不存在：{case_dir}')
    if observation:
        case.record_observation(observation, observation_data)
    _snapshot(case, new_data_dir)
    meta = json.loads(case.meta_path.read_text(encoding='utf-8'))
    meta['revision'] += 1
    meta['data_dir'] = str(new_data_dir)
    case.meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return case


def _snapshot(case: CaseFile, data_dir: str | Path) -> None:
    batches, scales, jobs, retains, slots, inquiries, empties = load_records(data_dir)
    analysis = analyze_case(batches, scales, jobs, retains, empties)
    plan = plan_isolation(analysis, slots)
    schedule = schedule_inquiries(analysis, inquiries, scales)
    case._append(EVENT_ANALYSIS, {
        'data_dir': str(data_dir),
        'analysis': _analysis_to_dict(analysis),
        'isolation_plan': _plan_to_dict(plan),
        'inquiry_schedule': {
            'selected': [
                {'id': opt.inquiry.id, 'value': opt.value, 'rationale': opt.rationale}
                for opt in schedule.selected
            ],
            'conflicts': schedule.conflicts,
            'options': {
                iid: {'value': opt.value, 'rationale': opt.rationale}
                for iid, opt in schedule.options.items()
            },
        },
    })
