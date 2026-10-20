import json
from utils.agent_schema_context import SchemaContextClient
from utils.agent_vlm import FingerprintedVLMClient


def test_schema_is_in_actual_prompt_and_fingerprint_without_mutating_sampling_or_schema(tmp_path):
    schema={'type':'object','properties':{'answer':{'type':'string','enum':['yes','unknown']}},'required':['answer']}
    client=FingerprintedVLMClient(base_url='http://unused',model='model',cache_dir=tmp_path)
    original=client.call_fingerprint(prompt='Task',image_paths=[],schema=schema);calls=[]
    client.call=lambda **kwargs:calls.append(kwargs) or kwargs
    wrapped=SchemaContextClient(client);wrapped.call(prompt='Task',schema=schema,image_paths=[],call_kind='test',shot_index=0,temporal_inputs=())
    new=calls[0]
    assert new['prompt'].startswith('Task\nRequired response JSON schema') and json.dumps(schema,ensure_ascii=False,sort_keys=True) in new['prompt']
    assert new['schema'] is schema and wrapped.cache_dir==tmp_path
    assert client.call_fingerprint(prompt=new['prompt'],image_paths=[],schema=schema)!=original
    assert client.sampling_config=={'temperature':0,'seed':0,'max_tokens':4096}
