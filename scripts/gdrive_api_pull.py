"""Download public Google Drive folders through the Drive v3 API.

Run from the project root: python scripts/gdrive_api_pull.py --once
The API key is read from G_DRIVE_KEY (environment or project .env).
"""

import argparse
import hashlib
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import random
import re
import socket
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse, parse_qs
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
API_URL = "https://www.googleapis.com/drive/v3/files"
FOLDER_MIME = "application/vnd.google-apps.folder"
GOOGLE_MIME = "application/vnd.google-apps."
DEFAULT_FOLDERS = {
    "a3": "12h4SSeCxbwxWME9hbdJ_Yv1pCXk-UabZ",
    "a1": "1cnJyHzMZnO8EyJJBep0gkDz1wMg1elXW",
}
def load_api_key():
    if os.environ.get("G_DRIVE_KEY", "").strip():
        return os.environ["G_DRIVE_KEY"].strip()
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            match = re.match(r"^\s*(?:export\s+)?G_DRIVE_KEY\s*=\s*(.*?)\s*$", line)
            if match:
                value = match.group(1).strip()
                if value.startswith(("'", '"')) and value.endswith(value[0]):
                    value = value[1:-1]
                else:
                    value = value.split(" #", 1)[0].strip()
                if value:
                    return value
    raise ValueError("缺少 G_DRIVE_KEY；請在專案 .env 或環境變數中設定。")


def parse_folder(value):
    name, separator, target = value.partition("=")
    if not separator or not name or not target or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise argparse.ArgumentTypeError("--folder 格式為 實驗名稱=資料夾ID或網址")
    parsed = urlparse(target)
    if parsed.scheme:
        match = re.search(r"/folders/([A-Za-z0-9_-]+)", parsed.path)
        if not match:
            raise argparse.ArgumentTypeError("資料夾網址須包含 /folders/ID")
        folder_id = match.group(1)
        resource_key = parse_qs(parsed.query).get("resourcekey", [None])[0]
    else:
        folder_id, _, resource_key = target.partition(":")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", folder_id):
        raise argparse.ArgumentTypeError("資料夾 ID 無效")
    if resource_key and not re.fullmatch(r"[A-Za-z0-9_-]+", resource_key):
        raise argparse.ArgumentTypeError("resource key 無效")
    return name, folder_id, resource_key or None


def safe_name(name):
    if (not name or name in {".", ".."} or name[-1] in {" ", "."}
            or re.search(r'[\\/:*?"<>|\x00-\x1f]', name)
            or re.fullmatch(r"(?i)(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", name)):
        raise ValueError(f"不安全的 Drive 檔名: {name!r}")
    return name


class DriveError(Exception):
    pass


class DriveClient:
    def __init__(self, api_key, *, retries=5, opener=urlopen, sleeper=time.sleep):
        self.api_key = api_key
        self.retries = retries
        self.opener = opener
        self.sleeper = sleeper

    def _request(self, url, *, resource_keys=(), stream=False):
        headers = {"X-Goog-Api-Key": self.api_key}
        keys = [f"{file_id}/{key}" for file_id, key in resource_keys if key]
        if keys:
            headers["X-Goog-Drive-Resource-Keys"] = ",".join(keys)
        for attempt in range(self.retries + 1):
            try:
                response = self.opener(Request(url, headers=headers), timeout=60)
                if stream:
                    return response
                with response:
                    return json.load(response)
            except HTTPError as exc:
                try:
                    payload = json.loads(exc.read(16384))
                    reason = payload.get("error", {}).get("errors", [{}])[0].get("reason", "")
                    message = payload.get("error", {}).get("message", str(exc.code))
                except (ValueError, KeyError, IndexError, AttributeError):
                    reason, message = "", str(exc.code)
                retry = exc.code in (429, 500, 502, 503, 504) or (
                    exc.code == 403 and reason in {"rateLimitExceeded", "userRateLimitExceeded"}
                )
                if not retry or attempt == self.retries:
                    raise DriveError(f"Drive API HTTP {exc.code} ({reason or message})") from None
            except (URLError, TimeoutError, socket.timeout) as exc:
                if attempt == self.retries:
                    raise DriveError(f"Drive 連線失敗: {exc.reason if isinstance(exc, URLError) else exc}") from None
            self.sleeper(min(60, 2 ** attempt + random.uniform(0, 1)))

    def list_children(self, folder_id, resource_key=None):
        token = None
        while True:
            params = {
                "q": f"'{folder_id}' in parents and trashed = false",
                "fields": "nextPageToken,files(id,name,mimeType,size,md5Checksum,modifiedTime,resourceKey)",
                "pageSize": 1000,
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            }
            if token:
                params["pageToken"] = token
            result = self._request(
                API_URL + "?" + urlencode(params),
                resource_keys=[(folder_id, resource_key)],
            )
            yield from result.get("files", [])
            token = result.get("nextPageToken")
            if not token:
                break

    def download(self, item, destination):
        file_id = item["id"]
        url = API_URL + "/" + quote(file_id, safe="") + "?alt=media&supportsAllDrives=true"
        temporary = destination.with_name(destination.name + ".gdrive-part")
        expected_size = int(item["size"]) if item.get("size") is not None else None
        expected_md5 = item.get("md5Checksum")
        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(self.retries + 1):
            try:
                with self._request(url, resource_keys=[(file_id, item.get("resourceKey"))], stream=True) as response:
                    digest = hashlib.md5()  # Drive exposes MD5 for binary files.
                    size = 0
                    with temporary.open("wb") as output:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            output.write(block)
                            digest.update(block)
                            size += len(block)
                if expected_size is not None and size != expected_size:
                    raise ValueError(f"下載大小不符 ({size}/{expected_size})")
                if expected_md5 and digest.hexdigest().lower() != expected_md5.lower():
                    raise ValueError("MD5 驗證失敗")
                if item.get("modifiedTime"):
                    from datetime import datetime
                    modified = datetime.fromisoformat(item["modifiedTime"].replace("Z", "+00:00")).timestamp()
                    os.utime(temporary, (modified, modified))
                os.replace(temporary, destination)
                return
            except (OSError, URLError, TimeoutError, socket.timeout, IncompleteRead, ValueError) as exc:
                if attempt == self.retries:
                    raise DriveError(f"{item['name']}: 下載失敗: {exc}") from None
                self.sleeper(min(60, 2 ** attempt + random.uniform(0, 1)))
            finally:
                temporary.unlink(missing_ok=True)


def unchanged(item, destination):
    if not destination.is_file():
        return False
    if item.get("size") is not None and destination.stat().st_size != int(item["size"]):
        return False
    if item.get("md5Checksum"):
        digest = hashlib.md5()
        with destination.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest().lower() == item["md5Checksum"].lower()
    if item.get("modifiedTime"):
        from datetime import datetime
        remote = datetime.fromisoformat(item["modifiedTime"].replace("Z", "+00:00")).timestamp()
        return abs(destination.stat().st_mtime - remote) < 2
    return False


def sync_folder(client, folder_id, output, *, resource_key=None, models_only=False, visited=None):
    visited = visited if visited is not None else set()
    if folder_id in visited:
        return 0
    visited.add(folder_id)
    count = 0
    errors = []
    for item in client.list_children(folder_id, resource_key):
        try:
            name = safe_name(item["name"])
            destination = output / name
            if item.get("mimeType") == FOLDER_MIME:
                if not models_only:
                    count += sync_folder(client, item["id"], destination,
                                         resource_key=item.get("resourceKey"), visited=visited)
                continue
            if item.get("mimeType", "").startswith(GOOGLE_MIME):
                print(f"略過 Google 文件 (需匯出): {name}")
                continue
            if models_only and not (re.fullmatch(r"G_.*\.pth", name) or
                                    re.fullmatch(r"config_.*\.yaml", name)):
                continue
            if unchanged(item, destination):
                continue
            print(f"下載 {destination.relative_to(ROOT) if destination.is_relative_to(ROOT) else destination}")
            client.download(item, destination)
            count += 1
        except (DriveError, ValueError, OSError, KeyError) as exc:
            print(f"檔案處理失敗: {exc}", file=sys.stderr)
            errors.append(str(exc))
    if errors:
        raise DriveError(f"{len(errors)} 個檔案或子資料夾處理失敗")
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description="Google Drive API 自動下載至本地 runs 暫存區")
    parser.add_argument("--folder", action="append", type=parse_folder,
                        help="實驗名稱=資料夾ID或分享網址，可重複指定；預設 a1/a3")
    parser.add_argument("--interval", type=int, default=300, help="輪詢間隔秒數 (預設 300)")
    parser.add_argument("--once", action="store_true", help="只執行一輪")
    parser.add_argument("--models-only", action="store_true", help="只同步 G_*.pth 和 config_*.yaml")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.interval < 1:
        parser.error("--interval 必須大於 0")
    folders = args.folder or [(name, folder_id, None) for name, folder_id in DEFAULT_FOLDERS.items()]
    try:
        client = DriveClient(load_api_key())
    except ValueError as exc:
        parser.error(str(exc))
    cycle = 1
    print("啟動 Google Drive API 同步；按 Ctrl+C 結束")
    try:
        while True:
            print(f"[{time.strftime('%H:%M:%S')}] 第 {cycle} 輪")
            failed = False
            for exp_name, folder_id, resource_key in folders:
                try:
                    count = sync_folder(client, folder_id, args.runs_dir / exp_name,
                                        resource_key=resource_key, models_only=args.models_only)
                    print(f"[{exp_name}] 完成，更新 {count} 個檔案")
                except (DriveError, OSError) as exc:
                    print(f"[{exp_name}] 同步失敗: {exc}", file=sys.stderr)
                    failed = True
            if args.once:
                return 1 if failed else 0
            cycle += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("已停止")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
