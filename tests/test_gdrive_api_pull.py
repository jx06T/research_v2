"""Offline behavior checks for the Drive API puller."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gdrive_api_pull.py"
spec = importlib.util.spec_from_file_location("gdrive_api_pull", SCRIPT)
pull = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pull)


class DrivePullTests(unittest.TestCase):
    def test_env_key_with_spaces_and_quotes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text('G_DRIVE_KEY = "test-key"\n', encoding="utf-8")
            with patch.object(pull, "ROOT", root), patch.dict(pull.os.environ, {}, clear=True):
                self.assertEqual(pull.load_api_key(), "test-key")

    def test_pagination_api_key_header_and_resource_key(self):
        requests = []

        def opener(request, timeout):
            requests.append(request)
            body = ({"files": [{"id": "one"}], "nextPageToken": "next"}
                    if len(requests) == 1 else {"files": [{"id": "two"}]})
            return io.BytesIO(json.dumps(body).encode())

        client = pull.DriveClient("private-test-key", opener=opener, sleeper=lambda _: None)
        self.assertEqual([item["id"] for item in client.list_children("folder", "rk")], ["one", "two"])
        self.assertTrue(all(req.get_header("X-goog-api-key") == "private-test-key" for req in requests))
        self.assertTrue(all(req.get_header("X-goog-drive-resource-keys") == "folder/rk" for req in requests))
        self.assertNotIn("private-test-key", requests[0].full_url)
        self.assertIn("pageToken=next", requests[1].full_url)

    def test_rate_limit_retried(self):
        attempts = []

        def opener(request, timeout):
            attempts.append(request)
            if len(attempts) == 1:
                body = b'{"error":{"errors":[{"reason":"rateLimitExceeded"}]}}'
                raise HTTPError(request.full_url, 403, "rate limited", {}, io.BytesIO(body))
            return io.BytesIO(b'{"files":[]}')

        delays = []
        client = pull.DriveClient("key", opener=opener, sleeper=delays.append)
        self.assertEqual(list(client.list_children("folder")), [])
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(delays), 1)

    def test_download_atomic_retry_and_skip(self):
        content = b"model-data"
        calls = []

        def opener(request, timeout):
            calls.append(request)
            return io.BytesIO(b"partial" if len(calls) == 1 else content)

        item = {"id": "file-id", "name": "G_run.pth", "size": len(content),
                "md5Checksum": hashlib.md5(content).hexdigest(), "resourceKey": "file-key"}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / item["name"]
            destination.write_bytes(b"old")
            client = pull.DriveClient("key", opener=opener, sleeper=lambda _: None)
            client.download(item, destination)
            self.assertEqual(destination.read_bytes(), content)
            self.assertFalse(destination.with_name(destination.name + ".gdrive-part").exists())
            self.assertTrue(pull.unchanged(item, destination))
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0].get_header("X-goog-drive-resource-keys"), "file-id/file-key")

    def test_sync_stays_in_experiment_staging_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            class Client:
                def list_children(self, folder_id, resource_key):
                    return [{"id": "file", "name": "G_run1.pth", "mimeType": "application/octet-stream"}]

                def download(self, item, destination):
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(b"weight")

            count = pull.sync_folder(Client(), "folder", root / "runs" / "a1")
            self.assertEqual(count, 1)
            self.assertEqual((root / "runs" / "a1" / "G_run1.pth").read_bytes(), b"weight")
            self.assertFalse((root / "registry.csv").exists())


if __name__ == "__main__":
    unittest.main()
