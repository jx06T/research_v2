"""Visual four-corner and grid-size calibration for one scanned page."""

import argparse
import io
import json
import os
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scans.calibration import suggest_grid, validate_calibration
from src.scans.extract import load_pages


PAGE = r'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<title>稿紙格線校正</title><style>
*{box-sizing:border-box}body{margin:0;font:16px system-ui,sans-serif;color:#1b2a2a;background:#eef3f0}
header{padding:14px 20px;background:#194b45;color:white}header strong{font-size:20px}header small{margin-left:14px;opacity:.85}
main{display:grid;grid-template-columns:310px minmax(450px,1fr);gap:14px;padding:14px;height:calc(100vh - 60px)}
.panel{background:#fff;border-radius:10px;padding:15px;box-shadow:0 2px 10px #0001;overflow:auto}
h2{font-size:17px;margin:0 0 9px}.help{font-size:13px;color:#536560;line-height:1.5;margin:5px 0 16px}
label{display:block;margin:10px 0 3px}input,button{font:inherit}input[type=number]{width:86px;padding:6px;border:1px solid #acbfba;border-radius:5px}
.counts{display:flex;gap:10px}.corner{display:flex;gap:8px;align-items:center;margin:8px 0}.corner b{width:44px}
.corner input{width:92px}.corner small{color:#5f6b68}
button{padding:9px 12px;border:1px solid #9cb8ae;background:#edf5f1;border-radius:6px;cursor:pointer}
button.primary{background:#1b7257;color:white;border-color:#1b7257}button.danger{color:#923636}
.actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}#notice{min-height:1.5em;margin-top:10px;color:#9b3030;white-space:pre-wrap}
.viewer{display:flex;flex-direction:column;min-width:0}.toolbar{display:flex;align-items:center;gap:12px;margin-bottom:10px}
.toolbar input{width:160px}.viewport{flex:1;overflow:auto;border:1px solid #bdccc7;background:#dce6e1;min-height:350px}
#stage{position:relative;margin:0 auto}#scan{display:block;width:100%;height:auto}#grid{position:absolute;left:0;top:0;width:100%;height:100%;touch-action:none;cursor:crosshair}
@media(max-width:900px){main{display:block;height:auto}.viewer{height:70vh;margin-top:14px}}
</style></head><body><header><strong>稿紙格線校正</strong><small id="source"></small></header>
<main><section class="panel"><h2>格線設定</h2><p class="help">拖曳預覽中的四個彩色角點，讓外框貼合稿紙最外側格線。可放大檢查，再調整欄列數；切格會等到你按「確認並切格」才開始。</p>
<div class="counts"><label>欄數<br><input id="cols" type="number" min="1" max="100"></label><label>列數<br><input id="rows" type="number" min="1" max="100"></label></div>
<h2 style="margin-top:18px">四角像素座標</h2><div id="corner-inputs"></div>
<p class="help" id="hint"></p><div class="actions"><button id="reset">恢復建議值</button><button class="primary" id="apply">確認並切格</button><button class="danger" id="cancel">取消</button></div><div id="notice"></div></section>
<section class="panel viewer"><div class="toolbar"><label>放大 <input id="zoom" type="range" min="1" max="3" step="0.1" value="1"></label><span id="zoom-label">100%</span></div>
<div class="viewport" id="viewport"><div id="stage"><img id="scan" src="/preview" alt="掃描頁預覽"><canvas id="grid"></canvas></div></div></section></main>
<script>
const $=id=>document.getElementById(id),names=['左上','右上','右下','左下'];
let cfg,corners=[],drag=-1,baseWidth=0,applied=false;
function fields(){const box=$('corner-inputs');box.innerHTML='';corners.forEach((p,i)=>{
 const row=document.createElement('div');row.className='corner';const title=document.createElement('b');title.textContent=names[i];row.append(title);
 for(let axis=0;axis<2;axis++){const input=document.createElement('input');input.type='number';input.step='0.1';input.value=p[axis].toFixed(1);input.title=(axis?'Y':'X')+' 像素';input.onchange=()=>{corners[i][axis]=Number(input.value);draw()};row.append(input)}box.append(row)})}
function solve(a,b){for(let i=0;i<8;i++){let pivot=i;for(let j=i+1;j<8;j++)if(Math.abs(a[j][i])>Math.abs(a[pivot][i]))pivot=j;
 [a[i],a[pivot]]=[a[pivot],a[i]];[b[i],b[pivot]]=[b[pivot],b[i]];if(Math.abs(a[i][i])<1e-10)return null;
 const d=a[i][i];for(let k=i;k<8;k++)a[i][k]/=d;b[i]/=d;
 for(let j=0;j<8;j++)if(j!==i){const f=a[j][i];for(let k=i;k<8;k++)a[j][k]-=f*a[i][k];b[j]-=f*b[i]}}return b}
function homography(){const uv=[[0,0],[1,0],[1,1],[0,1]],a=[],b=[];for(let i=0;i<4;i++){const [u,v]=uv[i],[x,y]=corners[i];
 a.push([u,v,1,0,0,0,-x*u,-x*v]);b.push(x);a.push([0,0,0,u,v,1,-y*u,-y*v]);b.push(y)}return solve(a,b)}
function draw(){if(!cfg||!$('scan').complete||!$('scan').naturalWidth)return;
 const img=$('scan'),canvas=$('grid'),w=img.clientWidth,h=img.clientHeight,dpr=devicePixelRatio||1;
 canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);canvas.style.width=w+'px';canvas.style.height=h+'px';
 const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,w,h);
 const sx=w/cfg.width,sy=h/cfg.height,H=homography();if(!H)return;
 const point=(u,v)=>{const den=H[6]*u+H[7]*v+1;return [(H[0]*u+H[1]*v+H[2])/den*sx,(H[3]*u+H[4]*v+H[5])/den*sy]};
 const cols=Number($('cols').value),rows=Number($('rows').value);if(!(cols>0&&rows>0&&cols<=100&&rows<=100))return;
 c.lineWidth=1;c.strokeStyle='rgba(245,65,38,.62)';for(let i=0;i<=cols;i++){const p=point(i/cols,0),q=point(i/cols,1);c.beginPath();c.moveTo(...p);c.lineTo(...q);c.stroke()}
 for(let j=0;j<=rows;j++){const p=point(0,j/rows),q=point(1,j/rows);c.beginPath();c.moveTo(...p);c.lineTo(...q);c.stroke()}
 c.font='bold 13px system-ui';corners.forEach((p,i)=>{const x=p[0]*sx,y=p[1]*sy;c.beginPath();c.arc(x,y,10,0,Math.PI*2);c.fillStyle=['#df3030','#1e76d3','#1a9659','#bb6c15'][i];c.fill();c.lineWidth=2;c.strokeStyle='white';c.stroke();c.lineWidth=3;c.strokeText(names[i],x+13,y-11);c.fillStyle='#173430';c.fillText(names[i],x+13,y-11)})}
function setZoom(){if(!cfg)return;const z=Number($('zoom').value);$('zoom-label').textContent=Math.round(z*100)+'%';$('stage').style.width=Math.round(baseWidth*z)+'px';requestAnimationFrame(draw)}
function pointer(event){const rect=$('grid').getBoundingClientRect();return [(event.clientX-rect.left)/rect.width*cfg.width,(event.clientY-rect.top)/rect.height*cfg.height]}
$('grid').onpointerdown=e=>{const p=pointer(e),scale=$('grid').clientWidth/cfg.width;let best=25/scale;for(let i=0;i<4;i++){
 const d=Math.hypot(corners[i][0]-p[0],corners[i][1]-p[1]);if(d<best){best=d;drag=i}}if(drag>=0)$('grid').setPointerCapture(e.pointerId)};
$('grid').onpointermove=e=>{if(drag<0)return;const p=pointer(e);corners[drag]=[Math.max(0,Math.min(cfg.width-1,Math.round(p[0]*10)/10)),Math.max(0,Math.min(cfg.height-1,Math.round(p[1]*10)/10))];fields();draw()};
$('grid').onpointerup=()=>drag=-1;$('grid').onpointercancel=()=>drag=-1;
$('zoom').oninput=setZoom;for(const id of ['cols','rows'])$(id).oninput=draw;
$('reset').onclick=()=>{corners=cfg.suggestion.corners.map(p=>p.slice());$('cols').value=cfg.suggestion.cols;$('rows').value=cfg.suggestion.rows;fields();draw()};
$('apply').onclick=async()=>{if(applied)return;const payload={corners,cols:Number($('cols').value),rows:Number($('rows').value)};
 try{const response=await fetch('/api/apply',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await response.json();
 if(!response.ok)throw Error(data.error||'儲存失敗');applied=true;$('notice').style.color='#176b4d';$('notice').textContent='設定已儲存，切格程序正在終端執行。';$('apply').disabled=true}
 catch(error){$('notice').textContent=error.message}};
$('cancel').onclick=async()=>{await fetch('/api/cancel',{method:'POST'});$('notice').textContent='已取消，未進行切格。'};
async function init(){cfg=await(await fetch('/api/config')).json();$('source').textContent=cfg.source;
 $('hint').textContent=`預覽 ${cfg.width} × ${cfg.height} 像素；偵測到 ${cfg.suggestion.detected_vertical_lines} 條直線、${cfg.suggestion.detected_horizontal_lines} 條橫線。建議欄列數請以畫面確認。`;
 corners=cfg.suggestion.corners.map(p=>p.slice());$('cols').value=cfg.suggestion.cols;$('rows').value=cfg.suggestion.rows;fields();
 $('scan').onload=()=>{baseWidth=Math.min($('viewport').clientWidth-2,cfg.width);setZoom()};if($('scan').complete&&$('scan').naturalWidth){baseWidth=Math.min($('viewport').clientWidth-2,cfg.width);setZoom()}
 new ResizeObserver(()=>{if(Number($('zoom').value)===1){baseWidth=Math.min($('viewport').clientWidth-2,cfg.width);setZoom()}else draw()}).observe($('viewport'))}
init();
</script></body></html>'''


def resolve_source(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"})
        if len(files) == 1:
            return files[0]
    raise ValueError("Calibration needs one PDF or image; select a single file from this directory")


def make_server(source: Path, output: Path, host: str, port: int,
                cols: int | None = None, rows: int | None = None) -> HTTPServer:
    source = resolve_source(source)
    pages = list(load_pages(source))
    if len(pages) != 1:
        raise ValueError("Visual calibration currently supports one-page files")
    image = pages[0][1]
    width, height = image.size
    suggestion = suggest_grid(image)
    if cols is not None:
        suggestion["cols"] = cols
    if rows is not None:
        suggestion["rows"] = rows
    preview = io.BytesIO()
    image.save(preview, format="PNG")
    preview_bytes = preview.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body: bytes, content_type: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value, status: int = 200):
            self.send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self):
            if self.path == "/":
                return self.send_bytes(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if self.path == "/preview":
                return self.send_bytes(preview_bytes, "image/png")
            if self.path == "/api/config":
                return self.send_json({"source": source.name, "width": width, "height": height, "suggestion": suggestion})
            return self.send_json({"error": "Not found"}, 404)

        def do_POST(self):
            if self.path == "/api/cancel":
                self.send_json({"cancelled": True})
                threading.Thread(target=server.shutdown, daemon=True).start()
                return
            if self.path != "/api/apply":
                return self.send_json({"error": "Not found"}, 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 10000:
                    raise ValueError("Invalid request size")
                payload = validate_calibration(json.loads(self.rfile.read(length)), width, height)
                if (output / "cells.csv").exists():
                    raise ValueError("cells.csv already exists; select a new output directory to avoid overwriting labels")
                record = {"source": str(source.resolve()), "page": 1, "rendered_size": [width, height],
                          **payload, "saved_utc": datetime.now(timezone.utc).isoformat()}
                output.mkdir(parents=True, exist_ok=True)
                temporary = output / "manual_calibration.tmp"
                temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(temporary, output / "manual_calibration.json")
                server.applied = True
                self.send_json({"saved": True})
                threading.Thread(target=server.shutdown, daemon=True).start()
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, 400)

    server = HTTPServer((host, port), Handler)
    server.applied = False
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cols", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = make_server(args.input, args.output, args.host, args.port, args.cols, args.rows)
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Adjust the grid at {url}; confirm to save and start extraction", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0 if server.applied else 2


if __name__ == "__main__":
    sys.exit(main())
