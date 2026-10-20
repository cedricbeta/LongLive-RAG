"""Quote/coordinate/parent/state contracts; synthetic assertions are not quality labels."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0,str(Path(__file__).parent))
from entity_verifier import project_frame_mask, NativeEntityVerifier, FrozenAttributeComparer
from utils.agent_script_entities import validate_compilation,entity_specs_at,repair_prompt
from utils.agent_edit import save_mask_payload
from utils.agent_runtime import ShotObservation,Verdict,file_sha256
from utils.agent_constraints import Constraint,ConstraintAssessment


@pytest.mark.parametrize('protocol',['typed_noun_phrases_v4','typed_candidates_v5','typed_name_candidates_v6'])
def test_native_verifier_refuses_unresolved_typed_input_before_models_or_outputs(tmp_path,protocol):
    with pytest.raises(ValueError,match='Unresolved typed'):
        NativeEntityVerifier({'compiler_protocol':protocol,'ready_for_development':False,'unresolved':[{'reason':'unknown modifier'}]},
            client=None,tracker=None,segmenter=None,work_dir=tmp_path/'not-created',grid=(2,2))
    assert not (tmp_path/'not-created').exists()


@pytest.mark.parametrize('start,count,expected',[(0,None,[0,1,2,3,4,5]),(2,1,[2]),(2,None,[2,3,4,5]),(5,1,[5])])
def test_requested_current_windows_keep_actual_authored_indices(start,count,expected):
    import importlib.util
    root=Path(__file__).resolve().parents[2]
    implementation=root/'implementation' if (root/'implementation').is_dir() else root
    spec=importlib.util.spec_from_file_location('native_window_test',implementation/'scripts/run_native_progressive.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.requested_shots(6,start,count)==expected


@pytest.mark.parametrize('start,count',[(-1,1),(6,1),(2,0),(2,5),(True,1),(0,False)])
def test_requested_current_windows_reject_invalid_ranges(start,count):
    import importlib.util
    root=Path(__file__).resolve().parents[2]
    implementation=root/'implementation' if (root/'implementation').is_dir() else root
    spec=importlib.util.spec_from_file_location('native_invalid_window_test',implementation/'scripts/run_native_progressive.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(ValueError,match='Invalid diagnostic'):
        module.requested_shots(6,start,count)


def test_resource_client_supports_actual_numbered_assignment_inputs_and_stops_calls(tmp_path):
    import importlib.util
    from PIL import Image
    from utils.agent_entity_judge import VLMLocalizer
    root=Path(__file__).resolve().parents[2]
    implementation=root/'implementation' if (root/'implementation').is_dir() else root
    spec=importlib.util.spec_from_file_location('native_runner_test',implementation/'scripts/run_native_progressive.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[]
    client=SimpleNamespace(cache_dir=tmp_path/'cache',call=lambda **kw:calls.append(kw))
    def deny():raise RuntimeError('resource_interrupted')
    wrapped=module.ResourceCheckedClient(client,deny)
    path=tmp_path/'frame.png';Image.new('RGB',(32,32),'white').save(path)
    output=VLMLocalizer(wrapped)._numbered_candidates(path,[(0,0,20,20)])
    assert output.exists() and output.parent==client.cache_dir/'assignment_inputs'
    with pytest.raises(RuntimeError,match='resource_interrupted'):wrapped.call(prompt='do not run')
    assert calls==[]


def compiled():
    prompts=['Iris meets Miles. Miles has blue hair.','Iris has red hair and wears a green coat. Miles leaves.']
    entities=[]
    for name,rows in [('Iris',[('hair','color','red',1),('garment','color','green',1),('garment','type','coat',1)]),
                      ('Miles',[('hair','color','blue',0)])]:
        entities.append({'name':name,'name_source':{'shot':0,'quote':name},
            'presence':[{'shot':k,'status':'required','quote':p} for k,p in enumerate(prompts)],
            'appearance':[{'part':p,'aspect':a,'expected':v,'source':{'shot':k,'quote':prompts[k]},'last_active_shot':1} for p,a,v,k in rows]})
    return prompts,{'entities':entities}


def test_exact_script_scope_prevents_future_traits_and_supports_multiple_entities():
    prompts,parsed=compiled();d=validate_compilation(prompts,parsed)
    s,r=entity_specs_at(d,0)
    assert [x.name for x in s]==['Iris','Miles'] and r['Iris']==[]
    assert [x['expected'] for x in r['Miles']]==['blue']
    _,r=entity_specs_at(d,1);assert len(r['Iris'])==3
    prompt=repair_prompt(prompts[1],'Iris',r['Iris'])
    assert prompt.startswith(prompts[1]+' ') and 'age' not in prompt and 'woman' not in prompt


@pytest.mark.parametrize('change',['invent_name','invent_value','future_activation','empty_presence','missing_presence','duplicate_name'])
def test_compiler_refuses_unbound_or_incomplete_sources(change):
    prompts,p=compiled()
    if change=='invent_name':p['entities'][0]['name']='Alice'
    if change=='invent_value':p['entities'][0]['appearance'][0]['expected']='auburn'
    if change=='future_activation':p['entities'][0]['appearance'][0]['last_active_shot']=0
    if change=='empty_presence':p['entities'][0]['presence'][0]['quote']=''
    if change=='missing_presence':p['entities'][0]['presence'].pop()
    if change=='duplicate_name':p['entities'].append(deepcopy(p['entities'][0]))
    with pytest.raises(ValueError):validate_compilation(prompts,p)


def test_compiler_refuses_conflicting_simultaneous_color():
    prompts,p=compiled();prompts[1]+=' A red and blue sign.'
    row=deepcopy(p['entities'][0]['appearance'][0]);row['expected']='blue';row['source']['quote']='blue'
    p['entities'][0]['appearance'].append(row)
    with pytest.raises(ValueError,match='Conflicting'):validate_compilation(prompts,p)


@pytest.fixture
def parent(tmp_path):
    p=tmp_path/'parent';(p/'videos').mkdir(parents=True);(p/'diagnostics').mkdir()
    (p/'videos/a.mp4').write_bytes(b'video')
    torch.save(torch.zeros(1,9,1,2,2),p/'diagnostics/latents_0000.pt')
    timing={'protocol':'continuous_vae_drop_first_frame_v1','export_frames':32,'segments':[
        {'start_export_frame':0,'export_frames':16,'start_latent_frame':0,'latent_frames':5},
        {'start_export_frame':16,'export_frames':16,'start_latent_frame':5,'latent_frames':4}]}
    (p/'diagnostics/timing_0000.json').write_text(json.dumps(timing))
    return p


def masks(parent,tmp_path,start=0):
    rows={i:torch.zeros(1,2,2) for i in range(start,start+16)}
    rows[start+1][0,0,0]=1;rows[start+3][0,1,0]=1;rows[start+15][0,0,1]=1
    path=save_mask_payload(rows,tmp_path/f'mask_{start}.pt',phrase='part',video=parent/'videos/a.mp4',grid=(2,2),frames_per_chunk=1)
    return path


def test_native_union_includes_all_four_frames_and_warmup_mapping(parent,tmp_path):
    mask=masks(parent,tmp_path)
    result=project_frame_mask(mask,parent=parent,segment=0,output=tmp_path/'native.pt')
    d=torch.load(result['path'],weights_only=True)
    assert d['mask'].shape==(5,2,2) and not d['mask'][0].any()
    assert d['mask'][1,0,0] and d['mask'][1,1,0] and d['mask'][4,0,1]
    assert d['source_frame_mapping'][1]['export_frames']==[0,1,2,3]


def test_native_second_segment_and_protection_union(parent,tmp_path):
    mask=masks(parent,tmp_path,16)
    rows={i:torch.zeros(1,2,2) for i in range(16,32)};rows[18][0,0,0]=1
    protect=save_mask_payload(rows,tmp_path/'protect.pt',phrase='other actor',video=parent/'videos/a.mp4',grid=(2,2),frames_per_chunk=1)
    r=project_frame_mask(mask,parent=parent,segment=1,output=tmp_path/'native.pt',protected=protect)
    d=torch.load(r['path'],weights_only=True)
    assert d['latent_range']==[5,9] and not d['mask'][0,0,0] and d['mask'][0,1,0]
    assert d['source_frame_mapping'][0]['export_frames']==[16,17,18,19]


@pytest.mark.parametrize('error',['video','range','nan','grid'])
def test_projection_refuses_stale_or_invalid_frame_masks(parent,tmp_path,error):
    mask=masks(parent,tmp_path);d=torch.load(mask,weights_only=True)
    if error=='video':d['video_sha256']='stale'
    if error=='range':d['masks'].pop(1)
    if error=='nan':d['masks'][1][0,0,0]=float('nan')
    if error=='grid':d['masks'][1]=torch.zeros(1,3,3)
    torch.save(d,mask)
    with pytest.raises(ValueError):project_frame_mask(mask,parent=parent,segment=0,output=tmp_path/'native.pt')


def test_wrapper_projects_options_and_quarantines_appearance_reference(parent,tmp_path):
    import yaml
    prompts,parsed=compiled();d=validate_compilation(prompts,parsed)
    (parent/'config.yaml').write_text(yaml.safe_dump({'timed_segments':[{'prompts':[p]} for p in prompts]}))
    (parent/'manifest.json').write_text(json.dumps({'config_sha256':file_sha256(parent/'config.yaml')}))
    (parent/'audit.json').write_text(json.dumps({'video_sha256':file_sha256(parent/'videos/a.mp4'),
        'latent_sha256':file_sha256(parent/'diagnostics/latents_0000.pt')}))
    mask=masks(parent,tmp_path,16)
    verifier=NativeEntityVerifier(d,client=None,tracker=None,segmenter=None,work_dir=tmp_path/'verifier',grid=(2,2))
    constraint=Constraint('a','attribute','red',subjects=('Iris',),shot_indices=(1,),expected='red',parameters={'part':'hair','aspect':'color'})
    metrics={'constraint_assessments':[ConstraintAssessment(constraint,'violated').to_dict()],
        'repair_options':[{'id':'attribute:Iris:hair','mask_sha256':file_sha256(mask),
            'action':{'tool':'edit_shot','params':{'shot':1,'entry_sigma':1.,'mask_path':str(mask),'prompt':'OLD'}}}]}
    def judge(observation,normative):
        assert observation.telemetry['chunk_start']==16 and observation.telemetry['chunk_end']==31
        verifier.judge_impl.reference_crops['Iris']=tmp_path/'untrusted.png'
        return Verdict(False,'fixture',metrics=metrics)
    verifier.judge_impl.judge=judge
    observation=ShotObservation(1,0,(),(),{'native_run':str(parent),'shot_video':str(parent/'videos/a.mp4')},(),'fingerprint')
    result=verifier.judge(observation,None)
    option=result.metrics['repair_options'][0]
    assert torch.load(option['action']['params']['mask_path'],weights_only=True)['latent_range']==[5,9]
    assert option['action']['params']['prompt'].startswith(prompts[1])
    assert not verifier.snapshot_state()['reference_crops'] and not result.metrics['repair_references']
    assert not result.directed_actions
    assert verifier.judge(observation,None)==result
    Path(option['action']['params']['mask_path']).write_bytes(b'mutated')
    with pytest.raises(ValueError,match='Cached'):verifier.judge(observation,None)


def test_missing_mention_cannot_erase_persistent_requirement():
    prompts,p=compiled();p['entities'][1]['appearance'][0]['last_active_shot']=0
    with pytest.raises(ValueError,match='must persist'):validate_compilation(prompts,p)


def test_later_explicit_scalar_change_ends_old_value():
    prompts,p=compiled();prompts[1]+=' Miles now has green hair.'
    p['entities'][1]['appearance'][0]['last_active_shot']=0
    p['entities'][1]['appearance'].append({'part':'hair','aspect':'color','expected':'green',
        'source':{'shot':1,'quote':'Miles now has green hair.'},'last_active_shot':1})
    d=validate_compilation(prompts,p)
    assert [r['expected'] for r in entity_specs_at(d,0)[1]['Miles']]==['blue']
    assert [r['expected'] for r in entity_specs_at(d,1)[1]['Miles']]==['green']


def test_compiler_has_at_most_one_program_validation_correction():
    from utils.agent_script_entities import compile_entities
    prompts,p=compiled();invalid=deepcopy(p);invalid['entities'][0]['appearance'][0]['expected']='invented'
    class Client:
        def __init__(self,rows):self.rows=rows;self.calls=[]
        def call(self,**kwargs):self.calls.append(kwargs);return {'parsed':self.rows[len(self.calls)-1]}
    client=Client([invalid,p]);d=compile_entities(client,prompts)
    assert d['compiler_attempts']==2 and len(client.calls)==2
    assert 'program_validation_error' in json.loads(client.calls[1]['prompt'])
    client=Client([invalid,invalid])
    with pytest.raises(ValueError):compile_entities(client,prompts)
    assert len(client.calls)==2


def test_isolated_quote_compiler_derives_persistence_and_prohibits_substring_traits():
    from utils.agent_script_entities import compile_entities_by_quotes
    prompts=['Iris, a young woman with red hair, walks.','Iris sits.']
    cast={'entities':[{'name':'Iris','name_source':{'shot':0,'quote':'Iris'},
        'presence':{'0':'required','1':'required'},'appearance_quotes':[{'shot':0,'quote':'a young woman with red hair'}]}]}
    rows=[{'part':'body','aspect':'age','expected':'young'}, {'part':'body','aspect':'type','expected':'woman'},
          {'part':'hair','aspect':'color','expected':'red'}]
    invalid=deepcopy(rows);invalid[1]['expected']='man'
    class Client:
        def __init__(self):self.results=[cast,{'requirements':invalid},{'requirements':rows}];self.calls=[]
        def call(self,**kwargs):self.calls.append(kwargs);return {'parsed':self.results[len(self.calls)-1]}
    c=Client();result=compile_entities_by_quotes(c,prompts)
    assert len(c.calls)==3 and all(row['last_active_shot']==1 for row in result['entities'][0]['appearance'])
    assert result['compiler_call_log'][1]['valid'] is False
    assert 'complete source words' in result['compiler_call_log'][1]['error']


def test_quote_compiler_does_not_silently_drop_unparsed_descriptors():
    from utils.agent_script_entities import compile_entities_by_quotes
    cast={'entities':[{'name':'Iris','name_source':{'shot':0,'quote':'Iris'},'presence':{'0':'required'},
        'appearance_quotes':[{'shot':0,'quote':'a young woman with red hair'}]}]}
    class Client:
        def __init__(self):self.count=0
        def call(self,**kwargs):
            self.count+=1
            return {'parsed':cast if self.count==1 else {'requirements':[{'part':'hair','aspect':'color','expected':'red'}]}}
    c=Client()
    with pytest.raises(ValueError,match='Uncovered'):compile_entities_by_quotes(c,['Iris, a young woman with red hair, walks.'])
    assert c.count==3
