#!/usr/bin/env python3
import hashlib, json, os, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

root = Path(__file__).resolve().parent
workspace = root.parents[2]
sys.path.insert(0, str(workspace))
from utils.agent_conditional_names import compile_conditional_name_entities
from utils.agent_schema_context import SchemaContextClient
from utils.agent_typed_script import require_typed_ready
from utils.agent_vlm import FingerprintedVLMClient
from utils.agent_gpu_ownership import require_owned_compute

read = lambda p: json.loads(p.read_text())
write = lambda p, value: p.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
plan, launch = read(root / 'plan.json'), read(root / 'model_launch.json')
budget = {'posts': 0}
started = time.monotonic()

class OneAttemptClient(FingerprintedVLMClient):
    def __init__(self, folder):
        super().__init__(base_url=launch['base_url'], model=plan['model'], cache_dir=folder / 'vlm_cache',
            timeout=plan['per_http_timeout_seconds'], max_parse_retries=0,
            max_tokens=plan['max_new_tokens'], seed=plan['seed'])
        self.folder, self.posts = folder, 0

    def _post(self, payload):
        require_owned_compute(subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader'], text=True), {launch['gpu_uuid']: {launch['server_pgid']}})
        assert self.posts < plan['maximum_posts_per_case'] and budget['posts'] < plan['maximum_http_posts']
        assert all(item['type'] == 'text' for item in payload['messages'][0]['content'])
        assert payload['response_format']['type'] == 'json_schema'
        self.posts += 1; budget['posts'] += 1
        before = time.monotonic()
        record = {'post': self.posts, 'global_post': budget['posts'], 'started_unix': time.time(),
            'request': payload, 'transport_retries': 0, 'schema_fallback': False}
        path = self.folder / f'http_{self.posts:02d}.json'
        write(path, record)
        try:
            request = urllib.request.Request(self.base_url + '/chat/completions',
                data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode())
                record.update(status='completed', http_status=response.status, response=raw)
            assert isinstance(raw, dict)
            return raw
        except urllib.error.HTTPError as error:
            record.update(status='failed', http_status=error.code, response_body=error.read().decode(errors='replace'))
            # Convert the error to prevent the inherited optional schema fallback.
            raise RuntimeError(f'HTTP {error.code}; this preflight forbids fallback or retry') from error
        except Exception as error:
            record.update(status='failed', error=type(error).__name__ + ': ' + str(error))
            raise
        finally:
            record['wall_seconds'] = time.monotonic() - before
            write(path, record)

rows = []
for case in plan['cases']:
    folder = root / 'cases' / case['id']; folder.mkdir(parents=True, exist_ok=False)
    before = time.monotonic(); client = OneAttemptClient(folder)
    row = {'case': case['id'], 'script_sha256': hashlib.sha256(json.dumps(case['prompts'], ensure_ascii=False).encode()).hexdigest()}
    try:
        compiled = compile_conditional_name_entities(SchemaContextClient(client), case['prompts'])
        write(folder / 'compiled_script.json', compiled)
        row.update(status='compiled', ready_for_development=compiled['ready_for_development'],
            unresolved=compiled['unresolved'], named_entities=[entity['name'] for entity in compiled['entities']],
            semantically_validated=compiled['semantically_validated'])
        try:
            require_typed_ready(compiled); row['readiness_gate'] = 'passed'
        except ValueError as error:
            row.update(readiness_gate='rejected', readiness_error=str(error))
    except Exception as error:
        row.update(status='execution_failed', error=type(error).__name__ + ': ' + str(error))
    row.update(wall_seconds=time.monotonic() - before, actual_http_posts=client.posts)
    write(folder / 'status.json', row); rows.append(row)
    write(root / 'case_results.json', rows)
    print(json.dumps(row), flush=True)
result = {'status': 'completed' if all(row['status'] == 'compiled' for row in rows) else 'completed_with_case_failures',
    'actual_http_posts': budget['posts'], 'worker_seconds': time.monotonic() - started,
    'cases': rows, 'new_video_generations': 0, 'visual_calls': 0, 'human_labels': 0,
    'semantic_accuracy_claim': False, 'repair_efficacy_claim': False}
write(root / 'result.json', result)
if result['status'] != 'completed': raise SystemExit(1)
