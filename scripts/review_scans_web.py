"""Local browser review for scanned glyphs; edits the cell manifest in place."""

import argparse
import json
import os
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scans.extract import read_manifest, write_manifest


PAGE = r'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<title>掃描字圖覆核</title><style>
body{margin:0;font:16px system-ui,sans-serif;color:#17212b;background:#f2f5f4}
header{display:flex;gap:16px;align-items:center;padding:12px 20px;background:#143f3b;color:white;flex-wrap:wrap}
main{display:grid;grid-template-columns:minmax(420px,1fr) minmax(350px,1fr);gap:18px;padding:18px;max-width:1500px;margin:auto}
.panel{background:white;border-radius:10px;padding:16px;box-shadow:0 2px 12px #0001}
.glyphs{display:flex;gap:18px;align-items:start;justify-content:center}.glyph{text-align:center;flex:1}
.glyph img{display:block;width:min(100%,280px);height:280px;object-fit:contain;border:1px solid #bacac6;background:white;margin:8px auto;image-rendering:auto}
.page-wrap{width:100%;max-height:75vh;overflow:auto}#page-inner{position:relative;width:100%}.page-wrap img{display:block;width:100%}
#mark{position:absolute;border:3px solid #e23838;box-sizing:border-box;pointer-events:none;background:#f003}
button,input,select{font:inherit;padding:8px 10px}button{cursor:pointer;border-radius:6px;border:1px solid #aac1b8;background:#eef6f2}
button.primary{background:#176c51;color:white;border-color:#176c51}button.danger{background:#8c3131;color:white;border-color:#8c3131}
.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:12px 0}.metadata{line-height:1.7;word-break:break-all;white-space:pre-line}
#label{width:90px;font-size:28px;text-align:center}#note{min-width:170px;flex:1}#notice{color:#8c3131;min-height:1.4em}
@media(max-width:950px){main{grid-template-columns:1fr}}
</style></head><body><header><strong>掃描字圖覆核</strong><label>顯示 <select id="filter"><option value="review">待覆核</option><option value="reviewed">已覆核</option><option value="all">全部</option><option value="disagree">Gemini／OCR 不同</option><option value="proposed">Gemini 候選</option><option value="unlabeled">未辨識</option><option value="model_rejected">模型未辨識</option><option value="human_rejected">人工拒絕</option><option value="accepted">已接受</option><option value="blank">空格</option></select></label><label>Gemini 分數 ≤ <select id="max-confidence"><option value="">全部</option><option value="3">3</option><option value="5">5</option><option value="7">7</option></select></label><label>排序 <select id="sort"><option value="reading">閱讀順序</option><option value="low-gemini">Gemini 低分優先</option><option value="low-ocr">OCR 低分優先</option><option value="disagreement">兩路不同優先</option></select></label><span id="summary"></span></header>
<main><section class="panel"><div class="controls"><button id="prev">← 上一格</button><span id="position"></span><button id="next">下一格 →</button></div>
<div class="metadata" id="meta"></div><div class="glyphs"><div class="glyph">原始裁切<img id="raw" alt="原始字圖"></div><div class="glyph">背景處理後<img id="clean" alt="清理字圖"></div></div>
<div class="controls"><label>正確字 <input id="label" autocomplete="off"></label><label>備註 <input id="note" autocomplete="off"></label></div>
<div class="controls"><button class="primary" id="accept">接受 Enter</button><button class="danger" id="reject">拒絕 R</button><button id="skip">跳過 →</button><button id="finish">完成覆核</button></div><div id="notice"></div>
<p>Enter 接受目前候選並跳到下一格；R 拒絕並跳到下一格；E 進入正確字輸入框，修改後按 Enter 接受；←／→ 切換格位。人工拒絕也算已覆核。分數只用於安排覆核順序，未人工檢查的高分字仍是候選。</p></section>
<section class="panel"><strong>所在頁面與格位</strong><div class="page-wrap"><div id="page-inner"><img id="page" alt="整頁稿紙"><div id="mark"></div></div></div></section></main>
<script>
let rows=[],visible=[],index=0,saving=false;const $=id=>document.getElementById(id);
const image=(id,kind)=>'/image/'+encodeURIComponent(id)+'/'+kind;
const reviewed=r=>r.label_source==='human_review'&&['accepted','rejected'].includes(r.status);
const disagree=r=>r.gemini_label&&r.ocr_label&&r.gemini_label!==r.ocr_label;
function filtered(){const f=$('filter').value,limit=$('max-confidence').value;
 const result=rows.filter(r=>(f==='all'||(f==='review'?(r.status!=='blank'&&!reviewed(r)):f==='reviewed'?reviewed(r):f==='disagree'?disagree(r):f==='human_rejected'?r.status==='rejected'&&reviewed(r):f==='model_rejected'?r.status==='rejected'&&!reviewed(r):r.status===f))&&(!limit||!r.gemini_confidence||Number(r.gemini_confidence)<=Number(limit)));
 const sort=$('sort').value;if(sort==='low-gemini')result.sort((a,b)=>Number(a.gemini_confidence||0)-Number(b.gemini_confidence||0));
 if(sort==='low-ocr')result.sort((a,b)=>Number(a.ocr_confidence||0)-Number(b.ocr_confidence||0));
 if(sort==='disagreement')result.sort((a,b)=>Number(Boolean(disagree(b)))-Number(Boolean(disagree(a))));
 return result}
function refresh(nextId){visible=filtered();const next=nextId?visible.findIndex(r=>r.id===nextId):-1;index=next>=0?next:Math.min(index,Math.max(visible.length-1,0));show()}
function show(){
 const done=rows.filter(reviewed).length,pending=rows.filter(r=>r.status!=='blank'&&!reviewed(r)).length;
 $('summary').textContent='全部 '+rows.length+'｜待覆核 '+pending+'｜已覆核 '+done+'（接受 '+rows.filter(r=>reviewed(r)&&r.status==='accepted').length+'、拒絕 '+rows.filter(r=>reviewed(r)&&r.status==='rejected').length+'）';
 const r=visible[index];$('position').textContent=r?`${index+1} / ${visible.length}`:'沒有符合條件的格位';
 if(!r){$('meta').textContent='';$('raw').removeAttribute('src');$('clean').removeAttribute('src');$('page').removeAttribute('src');return}
 const geminiScore=r.gemini_confidence?`${r.gemini_confidence}/10`:'未評分';
 const ocrScore=r.ocr_confidence?`${(Number(r.ocr_confidence)*100).toFixed(1)}%`:'無有效單字分數';
 const comparison=r.gemini_label&&r.ocr_label?(disagree(r)?'｜兩路不同':'｜兩路一致'):'';
 $('meta').textContent=`${r.id}　欄 ${r.col}／列 ${r.row}　狀態：${r.status}\nGemini：${r.gemini_label||'未辨識'}　自評 ${geminiScore}（${r.gemini_model||'—'}）\nOCR：${r.ocr_label||'無單字候選'}　模型分數 ${ocrScore}（${r.ocr_engine||'未執行'}；採用 ${r.ocr_image||'—'}）${comparison}\nOCR 原圖：${r.ocr_raw_text||'—'}（${r.ocr_raw_confidence||'—'}）　清理圖：${r.ocr_clean_text||'—'}（${r.ocr_clean_confidence||'—'}）`;
 $('raw').src=image(r.id,'raw');$('clean').src=image(r.id,'clean');$('page').src=image(r.id,'page');
 $('label').value=r.label||r.gemini_label||r.ocr_label||'';$('note').value=r.review_note||'';$('notice').textContent='';
 $('page').onload=()=>{const b=JSON.parse(r.bbox),w=$('page').naturalWidth,h=$('page').naturalHeight;
 const m=$('mark');m.style.left=(b[0]/w*100)+'%';m.style.top=(b[1]/h*100)+'%';m.style.width=((b[2]-b[0])/w*100)+'%';m.style.height=((b[3]-b[1])/h*100)+'%';
 document.querySelector('.page-wrap').scrollTop=((b[1]+b[3])/2/h)*$('page').clientHeight-document.querySelector('.page-wrap').clientHeight/2};
}
function move(delta){if(!visible.length)return;index=(index+delta+visible.length)%visible.length;show()}
async function save(decision){const r=visible[index];if(!r||saving)return;const label=$('label').value.trim();if(decision==='accept'&&[...label].length!==1){$('notice').textContent='接受時請輸入一個字；按 E 修改';return}
 const following=visible.slice(index+1).concat(visible.slice(0,index)).map(item=>item.id);saving=true;
 try{const resp=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:r.id,decision,label,note:$('note').value})});
 if(!resp.ok){$('notice').textContent=(await resp.json()).error||'儲存失敗';return}const updated=await resp.json();Object.assign(r,updated);document.activeElement.blur();const remaining=filtered();refresh(following.find(id=>remaining.some(item=>item.id===id)))}
 catch(error){$('notice').textContent='儲存失敗：'+error.message}finally{saving=false}}
async function init(){rows=await(await fetch('/api/rows')).json();refresh()}
for(const id of ['filter','max-confidence','sort'])$(id).onchange=()=>{index=0;refresh()};$('prev').onclick=()=>move(-1);$('next').onclick=$('skip').onclick=()=>move(1);
$('accept').onclick=()=>save('accept');$('reject').onclick=()=>save('reject');$('label').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();save('accept')}if(e.key==='Escape')e.target.blur()};
$('finish').onclick=async()=>{await fetch('/api/finish',{method:'POST'});$('notice').textContent='覆核已完成，可以關閉這個頁面。'};
document.onkeydown=e=>{if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName))return;
 if(e.key==='ArrowRight'){e.preventDefault();move(1)}else if(e.key==='ArrowLeft'){e.preventDefault();move(-1)}
 else if(e.key==='Enter'){e.preventDefault();save('accept')}else if(e.key.toLowerCase()==='r'){e.preventDefault();save('reject')}
 else if(e.key.toLowerCase()==='e'){e.preventDefault();$('label').focus();$('label').select()}};init();
</script></body></html>'''


def make_server(manifest: Path, host: str, port: int) -> HTTPServer:
    rows = read_manifest(manifest)
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate cell IDs in manifest")
    # Older UI saves did not mark a human rejection. Restore that distinction
    # from its audit trail so those decisions leave the pending queue.
    latest_review = {}
    audit = manifest.with_name("review_events.jsonl")
    if audit.is_file():
        with audit.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                latest_review[event.get("id")] = event.get("after", {}).get("status")
    restored = False
    for row in rows:
        if row["status"] == "rejected" and not row["label_source"] and latest_review.get(row["id"]) == "rejected":
            row["label_source"] = "human_review"
            restored = True
    if restored:
        temporary = manifest.with_suffix(".tmp")
        write_manifest(temporary, rows)
        os.replace(temporary, manifest)

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, payload: bytes, content_type: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, value, status=200):
            self.send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self.send_bytes(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/api/rows":
                return self.send_json(rows)
            parts = path.split("/")
            if len(parts) == 4 and parts[1] == "image":
                row = by_id.get(unquote(parts[2]))
                kind = parts[3]
                if not row or kind not in ("raw", "clean", "page"):
                    return self.send_json({"error": "Unknown image"}, 404)
                image = Path(row[f"{kind}_path"]) if kind != "page" else Path(row["raw_path"]).parents[1] / "rectified.png"
                if not image.is_file():
                    return self.send_json({"error": "Image missing"}, 404)
                return self.send_bytes(image.read_bytes(), "image/png")
            self.send_json({"error": "Not found"}, 404)

        def do_POST(self):
            if urlparse(self.path).path == "/api/finish":
                self.send_json({"finished": True})
                threading.Thread(target=server.shutdown, daemon=True).start()
                return
            if urlparse(self.path).path != "/api/review":
                return self.send_json({"error": "Not found"}, 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 10000:
                    raise ValueError("Request too large")
                payload = json.loads(self.rfile.read(length))
                row = by_id.get(payload.get("id"))
                if row is None:
                    raise ValueError("Unknown cell ID")
                before = {"status": row["status"], "label": row["label"], "label_source": row["label_source"]}
                decision = payload.get("decision")
                label = str(payload.get("label", "")).strip()
                if decision == "accept":
                    if len(label) != 1:
                        raise ValueError("Accepted label must be one character")
                    row["status"], row["label"], row["label_source"] = "accepted", label, "human_review"
                elif decision == "reject":
                    row["status"], row["label"], row["label_source"] = "rejected", "", "human_review"
                else:
                    raise ValueError("Decision must be accept or reject")
                row["review_note"] = str(payload.get("note", ""))[:500]
                temporary = manifest.with_suffix(".tmp")
                write_manifest(temporary, rows)
                os.replace(temporary, manifest)
                event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "id": row["id"],
                         "before": before, "after": {"status": row["status"], "label": row["label"],
                         "label_source": row["label_source"]}, "note": row["review_note"]}
                with manifest.with_name("review_events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                self.send_json(row)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, 400)

    server = HTTPServer((host, port), Handler)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    server = make_server(args.manifest, args.host, args.port)
    print(f"Review at http://{args.host}:{server.server_port}/ ; Ctrl+C stops the server")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
