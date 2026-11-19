"""Audit every assigned local-editor trial against its design, tool trace and qualified grade."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_trial(folder, trial, *, hosted=False):
    result, grade = (load(folder / name) for name in ('editor-result.json', 'grade.json'))
    keys = ('slug', 'arm', 'task_id', 'replicate', 'order')
    for key in keys if hosted else (*keys, 'seed'):
        if result[key] != trial[key] or grade[key] != trial[key]:
            raise ValueError(f"trial mismatch: {trial['slug']} {key}")
    events = [json.loads(line) for line in (folder / 'trajectory.jsonl').read_text().splitlines()]
    tool_events = []
    turn_number = 0
    for event in events:
        if event['type'] == 'updater':
            turn_number += 1
        elif event['type'] == 'tool':
            tool_events.append(dict(event, editor_turn=turn_number))
    succeeded = [e for e in tool_events if 'result' in e and 'error' not in e['result']]
    submitted = result['status'] == 'submitted'
    if submitted:
        sha = digest(folder / 'patch.safetensors')
        if not (sha == result['receipt']['sha256'] == load(folder / 'receipt.json')['sha256'] == grade['patch_sha256']):
            raise ValueError(f"patch digest mismatch: {trial['slug']}")
        if grade['episode_id'] != trial['task_id'] or grade['status'] not in ('scored', 'ineligible'):
            raise ValueError('unexpected qualified grade')
    elif result['status'] != 'no_submission' or grade['status'] != 'no_submission' or grade['reward'] != 0:
        raise ValueError('unexpected status for unsubmitted patch')
    if submitted != any(e['tool'] == 'submit_patch' for e in succeeded):
        raise ValueError('trace and editor receipt disagree')
    if submitted and grade['status'] == 'scored' and grade['reward'] != grade['metrics']['reward']:
        raise ValueError('qualified metric mismatch')
    measurements = [e for e in succeeded if e['tool'] == 'test_behavior' and 'after' in e['result']
                    and e['result']['identity']['matches']]
    recipes = [(e['editor_turn'], e['result']['intervention'].get('write')) for e in measurements]
    previews = [e for e in succeeded if e['tool'] == 'preview_weights']
    commits = {e['result']['candidate_id'] for e in succeeded if e['tool'] == 'commit_preview'}
    if not commits <= {e['result']['candidate_id'] for e in previews}:
        raise ValueError('commit without a matching preview')
    used = persisted = False
    for event in succeeded:
        direct = event['tool'] == 'write_weights'
        if not direct and event['tool'] != 'preview_weights':
            continue
        writes = [event['arguments']] if direct else event['arguments']['writes']
        exact = any(turn < event['editor_turn'] and recipe in writes for turn, recipe in recipes if recipe)
        used |= exact
        persisted |= exact and (direct or event['result']['candidate_id'] in commits)
    # This tracks the recipe reaching live weights. A later write can still overwrite its coordinates.
    totals = Counter()
    if hosted:
        generations = result['provider']
        reported = [r['usage'] for r in generations if r['status'] == 'completed']
        if reported != [e['usage'] for e in events if e['type'] == 'updater']:
            raise ValueError('provider and logged turn usage differ')
        for turn in generations:
            usage = turn['usage']
            if usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']:
                raise ValueError('provider token sum mismatch')
            totals.update({k: usage[k] for k in ('prompt_tokens','completion_tokens','total_tokens')})
        if abs(sum(t['usage']['cost_upper_bound_usd'] for t in generations) - result['usd_upper_bound']) > 0.000001:
            raise ValueError('provider and budget-ledger spend differ')
    else:
        generations = (folder / 'native-generations.jsonl').read_text().splitlines()
        for line in generations:
            turn = json.loads(line)
            if len(turn['generated_ids']) != turn['usage']['completion_tokens']:
                raise ValueError('generation length differs from counted output tokens')
            totals.update(turn['usage'])
    if dict(totals) != result['editor_usage'] or len(generations) != result['turns']:
        raise ValueError('editor token accounting differs from raw decoding')
    menu = {t['function']['name']: t['function']['parameters'] for t in load(folder / 'tools.json')}
    arm = trial['arm']
    if (('intervention' in menu['test_behavior']['properties']) != (arm in 'BD')
            or ('preview_weights' in menu) != (arm in 'CD')):
        raise ValueError('wrong tool interface')
    public_input = folder / 'public-input.json'
    metrics = grade['metrics'] or {}
    receipt = result['receipt'] or {}
    transitions = {
        name: dict(items=g['count'], retained=g['retained_successes'],
                   fixed=g['edited_successes']-g['retained_successes'],
                   regressed=g['base_successes']-g['retained_successes'])
        for name, g in metrics.get('groups', {}).items()
        if g and 'retained_successes' in g
    }
    return {**trial, 'grade': grade['status'], 'reward': grade['reward'], 'submitted': submitted,
            'qualified_transitions': transitions, 'elapsed_seconds': result['elapsed_seconds'],
            'changed_parameters': receipt.get('changed_parameters', 0), 'public_input_sha256': digest(public_input),
            'tool_menu_sha256': digest(folder / 'tools.json'), 'turns': result['turns'], **totals,
            'target_forwards': result['target_usage']['forwards'], 'target_forward_tokens': result['target_usage']['forward_tokens'],
            'successful_behavior_calls': sum(e['tool'] == 'test_behavior' for e in succeeded),
            'internal_requests': sum(e['tool'] == 'test_behavior' and 'intervention' in e.get('arguments', {}) for e in tool_events),
            'successful_internal_calls': len(measurements),
            'internal_changed_generation_ids': any(a['generation_id'] != b['generation_id'] for e in measurements
                                                   for a,b in zip(e['result']['baseline'], e['result']['after'], strict=True)),
            'internal_changed_visible_text': any(a['text'] != b['text'] for e in measurements
                                                 for a,b in zip(e['result']['baseline'], e['result']['after'], strict=True)),
            'internal_changed_exact_matches': any(a['correct'] != b['correct'] for e in measurements
                                                 for a,b in zip(e['result']['baseline'], e['result']['after'], strict=True)),
            'matching_recipe_attempted_successfully': used, 'matching_recipe_applied': persisted,
            'preview_requests': sum(e['tool'] == 'preview_weights' for e in tool_events),
            'successful_previews': len(previews), 'nonempty_previews': sum(e['result']['changed_parameters'] > 0 for e in previews),
            'successful_commits': len(commits), 'tool_errors': sum('error' in e or 'error' in e.get('result', {}) for e in tool_events),
            'target_gain': metrics.get('gain'), 'retained': metrics.get('controls_retained'),
            'retention_denominator': metrics.get('control_retention_denominator'), 'retention': metrics.get('control_retention')}


def summarize(design, trials_root, calibration_root):
    trials = [audit_trial(trials_root / t['slug'], t) for t in design['attempts']]
    tasks = [t['id'] for t in design['tasks']]
    if len({t['slug'] for t in trials}) != len(trials):
        raise ValueError('duplicate trial')
    for task in tasks:
        if len({t['public_input_sha256'] for t in trials if t['task_id'] == task}) != 1:
            raise ValueError('public task packet differs between arms')
    calibrations = {}
    for assignment in design['tasks']:
        calibration = load(calibration_root / (assignment['episode_sha256'] + '.json'))
        if calibration['id'] != assignment['id'] or calibration['episode_sha256'] != assignment['episode_sha256']:
            raise ValueError('calibration identity mismatch')
        if calibration['status'] not in ('eligible', 'ineligible'):
            raise ValueError('incomplete pristine calibration')
        calibrations[assignment['id']] = calibration['status']
    eligible = [task for task in tasks if calibrations[task] == 'eligible']
    for t in trials:
        if t['submitted'] and ((t['grade'] == 'scored') != (t['task_id'] in eligible)):
            raise ValueError('independent pristine calibration disagrees with grade')
    by_arm, by_task = {}, {}
    for arm in 'ABCD':
        rows = [t for t in trials if t['arm'] == arm]
        if len({t['tool_menu_sha256'] for t in rows}) != 1:
            raise ValueError('tool menu changed mid-collection')
        scored = [t for t in rows if t['grade'] == 'scored']
        paired_rows = [t for t in rows if t['task_id'] in eligible]
        denominator = sum(t['retention_denominator'] for t in scored)
        by_arm[arm] = dict(attempts=len(rows), eligible_attempts=len(paired_rows), submitted=sum(t['submitted'] for t in rows),
            nonempty=sum(t['changed_parameters'] > 0 for t in rows), scored=len(scored),
            mean_reward=mean(t['reward'] for t in paired_rows) if paired_rows else None,
            qualified_gain_on_scored=mean(t['target_gain'] for t in scored) if scored else None,
            qualified_retention_on_scored=sum(t['retained'] for t in scored)/denominator if denominator else None,
            retained=sum(t['retained'] for t in scored), retention_denominator=denominator,
            **{name: sum(bool(t[name]) for t in rows) for name in ('internal_requests','successful_behavior_calls','successful_internal_calls',
                'internal_changed_generation_ids','internal_changed_visible_text','internal_changed_exact_matches','matching_recipe_attempted_successfully',
                'matching_recipe_applied','preview_requests','successful_previews','successful_commits','tool_errors')},
            **{f'mean_{name}':mean(t[name] for t in rows) for name in ('total_tokens','prompt_tokens','completion_tokens','target_forwards','target_forward_tokens','elapsed_seconds')})
        slices = sorted({label for t in scored for label in t['qualified_transitions']})
        by_arm[arm]['qualified_transitions_on_scored'] = {
            label: dict(sum((Counter(t['qualified_transitions'].get(label, {})) for t in scored), Counter()))
            for label in slices}
        by_arm[arm]['call_totals'] = {
            name: sum(t[name] for t in rows) for name in ('internal_requests', 'successful_internal_calls',
                'successful_behavior_calls', 'preview_requests', 'successful_previews', 'successful_commits', 'tool_errors')}
    for task in eligible:
        by_task[task] = {arm: mean(t['reward'] for t in trials if t['arm']==arm and t['task_id']==task) for arm in 'ABCD'}
    contrasts = {}
    for label in ('D-C', 'B-A', 'C-A', 'D-B', 'D-A', '(D-C)-(B-A)'):
        vals = [r['D']-r['C']-r['B']+r['A'] if label.startswith('(') else r[label[0]]-r[label[2]] for r in by_task.values()]
        if vals:
            bootstrap = np.random.default_rng(29631).choice(vals,(20000,len(vals))).mean(axis=1)
            contrasts[label] = {'paired_mean':mean(vals),'task_bootstrap_95':np.percentile(bootstrap,[2.5,97.5]).tolist(),
                                'task_wins':sum(d>0 for d in vals),'task_ties':vals.count(0),'task_losses':sum(d<0 for d in vals)}
    return dict(assigned=len(trials), eligible_tasks=len(eligible), calibration=calibrations, arms=by_arm, task_means=by_task, contrasts=contrasts, trials=trials)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--trials', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(load(args.design), args.trials, args.calibration)
    report['design_sha256'] = digest(args.design)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
