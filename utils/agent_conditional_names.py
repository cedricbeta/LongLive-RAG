"""Opt-in name response schema with compatible kind/presence branches.

The candidate inventory, substantive instruction, owner stage and rejection
invariant are inherited unchanged. This constrains response structure, not
the correctness of the model's character or presence decisions.
"""
from copy import deepcopy

from utils.agent_name_candidates import compile_name_candidate_entities
from utils.agent_vlm import _validate_json


def conditional_name_schema(schema):
    result = deepcopy(schema)
    for key, decision in result['properties'].items():
        if decision['properties']['kind']['enum'] != ['character', 'noncharacter', 'ambiguous']:
            raise ValueError('Unexpected base name schema')
        ordinary = deepcopy(decision)
        ordinary['properties']['kind']['enum'] = ['character', 'ambiguous']
        noncharacter = deepcopy(decision)
        noncharacter['properties']['kind']['enum'] = ['noncharacter']
        for presence in noncharacter['properties']['presence']['properties'].values():
            presence['enum'] = ['not_required']
        result['properties'][key] = {'oneOf': [ordinary, noncharacter]}
    return result


class _ConditionalNameClient:
    def __init__(self, client):
        self.client = client

    def call(self, **kwargs):
        if kwargs.get('call_kind') != 'native_name_candidate_cast_v6':
            return self.client.call(**kwargs)
        updated = {**kwargs, 'schema': conditional_name_schema(kwargs['schema']),
                   'call_kind': 'native_name_candidate_cast_v7_conditional'}
        response = self.client.call(**updated)
        # This independent check still rejects contradictory replies from a
        # provider or fixture that ignores the requested output grammar.
        _validate_json(response['parsed'], updated['schema'])
        return response


def compile_conditional_name_entities(client, prompts):
    result = compile_name_candidate_entities(_ConditionalNameClient(client), prompts)
    result.update(compiler_protocol='typed_name_candidates_v7',
                  name_response_protocol='conditional_noncharacter_presence_v1')
    result['limits'].append('Kind/presence compatibility is structural only; no automatic pronoun classification, coreference or semantic-accuracy guarantee.')
    return result
