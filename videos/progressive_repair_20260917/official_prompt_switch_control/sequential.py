from pathlib import Path
import subprocess,json,time
root=Path(__file__).resolve().parent
for mode in ['off_gpu2','on_gpu2']:
 out=root/mode
 with (out/'supervisor.log').open('x') as log:
  p=subprocess.Popen(['python',str(root/'supervisor.py'),str(out)],stdout=log,stderr=subprocess.STDOUT)
  (out/'supervisor.pid').write_text(str(p.pid)+'\n')
  code=p.wait()
 if code!=0: raise SystemExit(code)
 status=json.loads((out/'status.json').read_text())
 if status['status']!='completed': raise SystemExit(1)
 # The child has exited; a short driver teardown grace is not a retry.
 time.sleep(3)
