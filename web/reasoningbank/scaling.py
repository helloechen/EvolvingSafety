"""Optional MaTTS orchestration adapted from V6, independent of its experiment policy.

Rollout callbacks own environment resets/isolation, scoring and artifact storage.
Reflection callbacks own induction and persistence. Parallel mode means independent
samples with frozen memory; it does not start concurrent browser sessions.
"""
from copy import deepcopy


def run_matts(*, rollout, retrieve=None, reflect=None, mode='none', k=1,
              learn_from='all', stop_on_success=True):
    """Run optional scaling with explicit callbacks.

    rollout(memory, index) -> scored record containing a boolean success.
    retrieve(include_current) -> memory snapshot for a rollout.
    reflect(records) -> persist experience from the selected scored records.
    The default executes one ordinary rollout; all scaling is opt-in.
    """
    if mode not in {'none', 'seq', 'parallel'}:
        raise ValueError('mode must be none, seq or parallel')
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError('k must be a positive integer')
    if mode == 'none' and k != 1:
        raise ValueError('Enable a scaling mode explicitly to use k > 1')
    if learn_from not in {'all', 'success', 'failure', 'none'}:
        raise ValueError('learn_from must be all, success, failure or none')

    def eligible(record):
        return (learn_from == 'all' or
                learn_from == ('success' if record['success'] else 'failure'))

    frozen = retrieve(False) if mode == 'parallel' and retrieve else ''
    records = []
    for index in range(1, k + 1):
        memory = (deepcopy(frozen) if mode == 'parallel' else
                  retrieve(index > 1) if retrieve else '')
        record = rollout(memory, index)
        if not isinstance(record, dict) or type(record.get('success')) is not bool:
            raise ValueError('Every rollout requires an explicit boolean success result')
        status = record.get('evaluation_status', 'success' if record['success'] else 'failure')
        if status not in {'success', 'failure'} or (status == 'success') != record['success']:
            raise ValueError('Unscored/error rollout cannot be aggregated as a task outcome')
        if any(record.get(key) for key in ['protocol_error', 'agent_error', 'evaluator_error', 'infrastructure_error']):
            raise ValueError('Technical errors cannot be aggregated as task outcomes')
        record = deepcopy(record)
        record['rollout'] = index
        records.append(record)
        if mode != 'parallel' and reflect and eligible(record):
            reflect([deepcopy(record)])
        if mode == 'seq' and stop_on_success and record['success']:
            break
    if mode == 'parallel' and reflect:
        selected = [deepcopy(record) for record in records if eligible(record)]
        if selected:
            reflect(selected)
    return {'mode': mode, 'budget': k, 'learn_from': learn_from,
            'stop_on_success': stop_on_success,
            'success': any(record['success'] for record in records),
            'rollouts': records}
