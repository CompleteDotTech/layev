"""Export machine-readable contracts; Python enforces additional cross-field invariants."""
from dataclasses import fields
from pathlib import Path
import json
from pydantic import TypeAdapter
from kev_laya.schema import SystemOneResponse
from kev_laya.model import BackboneConfig
from kev_laya.execution import BatchPolicy, EXECUTION_VERSION
from kev_laya.training import TrainSettings
from kev_laya.objectives import ObjectiveConfig
from kev_laya.telemetry_contract import PHASES,METRICS,V2_METRICS,MAX_HISTORY,MAX_ARTIFACTS

root=Path(__file__).resolve().parents[1]/'schemas';root.mkdir(exist_ok=True)
def write(name,obj):
    obj={'$schema':'https://json-schema.org/draft/2020-12/schema',**obj}
    (root/name).write_text(json.dumps(obj,indent=2)+'\n')
def obj(properties,required=None):return {'type':'object','properties':properties,'required':list(properties) if required is None else required,'additionalProperties':False}
def integer(nullable=False):return {'type':['integer','null'] if nullable else 'integer','minimum':0}
def number(nullable=False):return {'type':['number','null'] if nullable else 'number','minimum':0}
def nullable(schema):return {'anyOf':[schema,{'type':'null'}]}
entry={'type':['string','object','array','null']}
question={'oneOf':[
    obj({'type':{'const':'choice'},'instructions':entry,'criteria':{'type':'object','minProperties':1,'maxProperties':255,'propertyNames':{'minLength':1,'maxLength':512},'additionalProperties':entry}}),
    obj({'type':{'const':'score'},'instructions':entry,'criteria':{'type':'array','minItems':2,'maxItems':10,'items':entry}}),
    obj({'type':{'const':'noul'},'instructions':entry,'criteria':nullable(obj({'true':entry,'false':entry},[]))},['type','instructions'])]}
write('api-request.schema.json',obj({'state':{'type':['string','object','array']},'model':{'type':'string','minLength':1},
    'questions':{'type':'object','minProperties':1,'maxProperties':1024,'propertyNames':{'minLength':1,'maxLength':512},'additionalProperties':question}}))
write('api-response.schema.json',SystemOneResponse.model_json_schema())
write('train-config.schema.json',obj({'backbone':TypeAdapter(BackboneConfig).json_schema(),'training':TypeAdapter(TrainSettings).json_schema(),
    'execution':{**TypeAdapter(BatchPolicy).json_schema(), 'additionalProperties':False,
        'properties':TypeAdapter(BatchPolicy).json_schema()['properties']|{'version':{'const':EXECUTION_VERSION}}},
    'objective':TypeAdapter(ObjectiveConfig).json_schema(),'limits':obj({'branch':{'type':'integer','minimum':8},'aggregate':{'type':'integer','minimum':8},'max_questions':{'type':'integer','minimum':1}},['branch','aggregate'])},['training','objective','limits']))
identity={'type':'string','pattern':r'^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,159}$'}
sha={'type':'string','pattern':'^[a-f0-9]{64}$'}
date={'type':'string','format':'date-time'}
uri={'type':'string','format':'uri','maxLength':2048}
metric={'type':'object','additionalProperties':False,'properties':{m:{'type':'number'} for m in sorted(METRICS)}}
counters={k:integer() for k in ['optimizer_steps','microbatches','examples','forward_tokens']}
provenance=obj({**{k:{'type':'string','maxLength':240} for k in ['model','backbone','backbone_revision','tokenizer','precision','hardware','evidence_class']},
    'config_sha256':sha,'data_sha256':sha,'split_hashes':obj({k:sha for k in ['train','development','calibration','test']},[]),
    'seed':integer(),'repository':nullable(uri),'commit':nullable({'type':'string','pattern':'^[a-f0-9]{40,64}$'}),
    'context_limits':obj({'branch':{'type':'integer','minimum':1},'aggregate':{'type':'integer','minimum':1}})})
properties={
    'schema_version':{'const':1},'framework':{'const':'kev_laya'},'framework_version':identity,
    **{k:identity for k in ['experiment_id','run_id','attempt_id']},'attempt_index':integer(),'parent_attempt_id':nullable(identity),
    'sequence':integer(),**{k:date for k in ['started_at','updated_at','heartbeat_at']},'phase':{'enum':sorted(PHASES)},
    'scheduler':nullable(obj({'provider':{'const':'skypilot'},'namespace':identity,'job_id':integer()})),
    'wandb':nullable(obj({k:identity for k in ['entity','project','run_id']})),
    'progress':obj(counters|{'epoch':number(True),'elapsed_seconds':number(True),'eta_seconds':number(True),'tokens_per_second':number(True),
                           'totals':obj({k:integer(True) for k in counters},[])}),
    'metrics':metric,'history':{'type':'array','maxItems':MAX_HISTORY,'items':obj({'step':integer(),'phase':{'enum':sorted(PHASES)},'metrics':metric})},
    'provenance':provenance,
    'artifacts':{'type':'array','maxItems':MAX_ARTIFACTS,'items':obj({'kind':{'enum':['checkpoint','configuration','evaluation','manifest']},
        'uri':uri,'sha256':sha,'size_bytes':integer(),'parent_sha256':nullable(sha),'resumable':{'type':'boolean'}})},
    'serving':obj({k:integer() for k in ['requests','errors','input_tokens','forward_tokens','output_tokens','questions','prefix_reuses']}|
                 {'latency_ms':{'type':'array','maxItems':MAX_HISTORY,'items':number()}}),
    'resume':obj({'capable':{'type':'boolean'},'checkpoint_sha256':nullable(sha)}),
    'recoveries':obj({k:integer(True) for k in ['total','infrastructure','application']}),
    'cost':obj({'measured_usd':number(True),'estimated_usd':number(True),'basis':{'enum':[None,'provider_billing','configured_hourly_rate','metered_local']}}),
    'monitoring_export_failures':integer()}
write('telemetry.schema.json',obj(properties)|{'description':'Structural v1 schema. The standalone Python contract additionally enforces finite values, timestamp ordering, byte bounds, credential-free URIs and lineage/counter consistency.'})
write('registry.schema.json',obj({'schema_version':{'const':1},'sources':{'type':'array','maxItems':512,'items':{'oneOf':[
    obj({'id':identity,'transport':{'const':'local'},'path':{'type':'string'}}),
    obj({'id':identity,'transport':{'const':'wandb'},'entity':identity,'project':identity,'limit':{'type':'integer','minimum':1,'maximum':512}}),
    obj({'id':identity,'transport':{'const':'s3'},'bucket':{'type':'string'},'key':{'type':'string'}})]}}}))
print(root)
# v2 is a deliberate extension; retain strict v1 historical validation.
import copy
v1 = obj(copy.deepcopy(properties))
write('telemetry-v1.schema.json', v1)
v2_props = copy.deepcopy(properties)
v2_props['schema_version'] = {'const': 2}
v2_metric = {'type':'object','additionalProperties':False,'properties':{m:{'type':'number'} for m in sorted(V2_METRICS)}}
v2_props['metrics'] = v2_metric
v2_props['history']['items']['properties']['metrics'] = v2_metric
v2_props['framework_version'] = {'type':'string','pattern':r'^[A-Za-z0-9][A-Za-z0-9_.+:/-]{0,159}$'}
v2_props['provenance']['properties']['config_sha256'] = nullable(sha)
v2_props['provenance']['properties']['data_sha256'] = nullable(sha)
attempt = obj({'experiment_id': identity, 'run_id': identity, 'attempt_id': identity, 'attempt_index': integer(),
               'parent_attempt_id': nullable(identity), 'previous_sha256': nullable(sha), 'sha256': sha})
execution = obj({k: integer() for k in ('prefix_passes','branch_passes','compute_tokens','padding_tokens','max_batch_size','useful_forward_tokens','questions')} |
                {'batch_size_histogram': {'type': 'object', 'maxProperties':1024, 'propertyNames': {'pattern':'^[0-9]+$'}, 'additionalProperties': integer()}})
resource = obj({'sampled_at': date, 'interval_seconds': {'type':'number','minimum':.001}, 'units': {'const':'bytes'},
                **{k: integer(True) for k in ('rss_bytes','gpu_allocated_bytes','gpu_peak_allocated_bytes')},
                'unavailable': {'type':'array','items':{'enum':['rss','cuda']}}})
calibration = obj({'status': {'type':'string','maxLength':80}, 'method':nullable({'type':'string','maxLength':80}),
    'split_sha256':nullable(sha), 'per_type':obj({k:obj({'temperature':{'type':'number','minimum':.2,'maximum':5},
        'count':integer(), 'status':{'type':'string','maxLength':80}, 'before_nll':number(True),'after_nll':number(True)})
        for k in ('choice','score','noul')},[])})
v2_props['extensions'] = obj({
    'serialization': obj({k: {'type':'string','maxLength':240} for k in ('tokenizer','serialization','literal_encoding')}),
    'source': obj({'source_tree_sha256': nullable(sha), 'archive_sha256': nullable(sha), 'git_dirty': {'type':['boolean','null']},
                   'git_status': {'enum':['verified','unavailable','error']}, 'package_version': {'type':'string','maxLength':80}}),
    'attempt_lineage': {'type':'array','maxItems':64,'items':attempt}, 'lineage_durable':nullable({'type':'boolean'}), 'execution': nullable(execution), 'resources':nullable(resource), 'calibration':nullable(calibration)},
    ['serialization','source','attempt_lineage','lineage_durable','execution','resources'])
write('telemetry.schema.json', {'oneOf':[v1, obj(v2_props)],
    'description':'Historical strict v1 or extended strict v2. Python additionally validates receipt hashes, parent identity, timestamps, credential-free URIs, resource units and accounting invariants.'})
