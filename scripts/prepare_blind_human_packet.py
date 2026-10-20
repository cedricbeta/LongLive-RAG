#!/usr/bin/env python3
"""Build a local anonymous A/B development packet; never create human labels."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import time
import zipfile

HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Video comparison pilot</title>
<style>body{font:16px system-ui;max-width:1400px;margin:24px auto;padding:0 18px;background:#f6f7f9;color:#17212b}button,select,textarea{font:inherit;padding:9px;margin:5px}video{width:100%;background:#000}section{background:white;border:1px solid #ccd2da;border-radius:9px;padding:18px;margin:18px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}h1{font-size:26px}h2{font-size:20px}.criterion{display:grid;grid-template-columns:minmax(240px,1fr) 250px;align-items:center;border-top:1px solid #ddd;padding:7px}textarea{width:90%;min-height:65px}summary{cursor:pointer}li{margin:8px 0}.hint{color:#536171}@media(max-width:700px){.pair,.criterion{grid-template-columns:1fr}}</style>
<h1>Anonymous video comparison · development pilot</h1>
<p>Watch both versions with the same six-part script. Compare only what is visible. A script requirement is not proof that it appears in either video. Neither side is assumed to be correct. You can replay, pause and seek.</p>
<p>Choose <b>Cannot judge</b> for hidden faces, insufficient evidence or uncertain differences; choose <b>About equal</b> when you can judge and see no meaningful difference. Consider the whole video, including later appearances and required interactions. Better clothing does not cancel a changed person or lost action.</p>
<p class="hint">No name, email or login is requested. Your anonymous session code and ratings stay in this browser until you download the JSON file. Nothing is uploaded. Return that file through the evaluation coordinator's agreed channel. This pilot is not the final benchmark study.</p>
<div id="cases"></div><button id="download">Download completed ratings</button><p id="status" role="status"></p>
<script src="cases.js"></script><script>
const criteria=[['appearance','Which version better follows the explicit appearance and scene requirements?'],['identity','Which better preserves each depicted person across time?'],['actions','Which better follows the required movements and interactions?'],['quality','Which has fewer distracting visual defects?'],['overall','Considering all requirements and damage, which is better overall?']];
const choices=[['','Choose a rating'],['a','Version A is better'],['b','Version B is better'],['tie','About equal'],['unknown','Cannot judge']];
const root=document.getElementById('cases');const sessionId=crypto.randomUUID?crypto.randomUUID():Date.now().toString(36)+Math.random().toString(36).slice(2);const opened=new Date().toISOString();
for(const item of CASES){const section=document.createElement('section');section.dataset.id=item.id;
const title=document.createElement('h2');title.textContent=item.id;section.append(title);
const details=document.createElement('details');details.open=true;const summary=document.createElement('summary');summary.textContent='Authored script and timing';details.append(summary);const list=document.createElement('ol');
for(const row of item.script){const li=document.createElement('li');li.textContent=row.start_seconds+'–'+(row.start_seconds+row.seconds)+' s: '+row.prompt;list.append(li)}details.append(list);section.append(details);
const pair=document.createElement('div');pair.className='pair';const videos=[];for(const side of ['a','b']){const panel=document.createElement('div');const label=document.createElement('h3');label.textContent='Version '+side.toUpperCase();const video=document.createElement('video');video.controls=true;video.preload='metadata';video.src=item[side];video.muted=true;panel.append(label,video);pair.append(panel);videos.push(video)}section.append(pair);
for(const [label,action] of [['Play both',()=>{videos[1].currentTime=videos[0].currentTime;videos.forEach(v=>v.play())}],['Pause both',()=>videos.forEach(v=>v.pause())],['Restart both',()=>videos.forEach(v=>{v.pause();v.currentTime=0})]]){const b=document.createElement('button');b.textContent=label;b.onclick=action;section.append(b)}
for(const [key,label] of criteria){const row=document.createElement('label');row.className='criterion';row.append(document.createTextNode(label));const select=document.createElement('select');select.dataset.criterion=key;for(const [value,text] of choices){const option=document.createElement('option');option.value=value;option.textContent=text;select.append(option)}row.append(select);section.append(row)}
const note=document.createElement('textarea');note.placeholder='Optional: visible differences and timestamps. Do not include personal information.';section.append(note);root.append(section)}
document.getElementById('download').onclick=()=>{const ratings=[];for(const section of root.querySelectorAll('section')){const answers={};for(const select of section.querySelectorAll('select')){if(!select.value){document.getElementById('status').textContent='Please rate every item, using Cannot judge when needed.';select.focus();return}answers[select.dataset.criterion]=select.value}ratings.push({case_id:section.dataset.id,answers,notes:section.querySelector('textarea').value})}
const data={schema_version:1,packet_id:PACKET_ID,anonymous_session_id:sessionId,opened_at:opened,submitted_at:new Date().toISOString(),ratings};const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download='ratings_'+sessionId+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);document.getElementById('status').textContent='Ratings downloaded locally. No data were sent.'};
</script></html>'''


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(manifest,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    source=json.loads(Path(manifest).read_text());packet=output/'participant_packet';packet.mkdir();(packet/'media').mkdir()
    public=[];private=[];packet_id=secrets.token_hex(12)
    rows=list(source['cases']);secrets.SystemRandom().shuffle(rows)
    for index,row in enumerate(rows):
        side=bool(secrets.randbits(1));paths=[Path(row['parent']),Path(row['candidate'])]
        source_info=[]
        for path in paths:
            result=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=width,height,r_frame_rate,duration,nb_frames','-of','json',str(path)],text=True))['streams'][0]
            source_info.append(result)
        if source_info[0]!=source_info[1]:raise ValueError('Paired human videos must have equal duration/fps/shape')
        if side:paths.reverse()
        identifier=f'Pair {index+1:02d}';items={}
        for label,path in zip(('a','b'),paths):
            name=f'media/pair{index+1:02d}_{label}.mp4';shutil.copy2(path,packet/name);items[label]=name
        public.append({'id':identifier,**items,'script':row['script']})
        private.append({'case_id':identifier,'source_case':row['id'],'a_is_candidate':side,
            'parent_sha256':sha(row['parent']),'candidate_sha256':sha(row['candidate']),
            'source_video_info':source_info,'script':row['script'],'source_role':'diagnostic candidate, not necessarily delivered output'})
    key={'packet_id':packet_id,'private_nonce':secrets.token_hex(32),'assignments':private}
    (output/'private_assignment_key.json').write_text(json.dumps(key,indent=2)+'\n')
    (packet/'index.html').write_text(HTML)
    (packet/'cases.js').write_text('const PACKET_ID='+json.dumps(packet_id)+';\nconst CASES='+json.dumps(public,ensure_ascii=False)+';\n')
    (packet/'README.txt').write_text('Open index.html in a browser with media files beside it. If local media is blocked, run python -m http.server 8000 in this folder and open http://127.0.0.1:8000. Do not distribute the private assignment key to raters. No example ratings or model judgments are included.\n')
    with zipfile.ZipFile(output/'participant_packet.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(packet.rglob('*')):
            if file.is_file():archive.write(file,file.relative_to(output))
    result={'packet_id':packet_id,'case_count':len(public),'source_manifest_sha256':sha(manifest),
        'assignment_key_commitment_sha256':sha(output/'private_assignment_key.json'),
        'participant_zip_sha256':sha(output/'participant_packet.zip'),'files':{str(p.relative_to(packet)):sha(p) for p in packet.rglob('*') if p.is_file()},
        'created_unix':time.time(),'actual_human_labels':0,'raters_recruited':0,
        'source_sha256':sha(__file__),'scope':'Two exposed native diagnostic pairs for human evaluation pilot; not formal independent test statistics.',
        'private_key_policy':'Keep assignment key outside the distributed packet until rating collection is closed.'}
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps(build(args.manifest,args.output),indent=2))
