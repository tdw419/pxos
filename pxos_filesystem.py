import os
import pickle
import json

class PxOSFilesystem:
    def __init__(self, persist_path="pxos_fs.bin", version_path="versions.json"):
        self.files = {}
        self.names = {}
        self.next_handle = 1
        self.persist_path = persist_path
        self.version_path = version_path
        self.version_db = {}
        self.load()

    def open(self, filename_id, mode):
        name = self.names.get(filename_id, f"file_{filename_id}")
        handle = self.next_handle
        self.next_handle += 1

        if mode == 2 or name not in {f["name"] for f in self.files.values()}:
            self.files[handle] = {"name": name, "data": bytearray()}
        else:
            for h, f in self.files.items():
                if f["name"] == name:
                    handle = h
                    break
        return handle

    def write(self, handle, data):
        if handle in self.files:
            self.files[handle]["data"].extend(data)
            self.save()
            return len(data)
        return 0

    def read(self, handle, maxlen):
        if handle in self.files:
            data = bytes(self.files[handle]["data"][:maxlen])
            return data
        return b""

    def close(self, handle):
        return 0

    def save(self):
        with open(self.persist_path, "wb") as f:
            pickle.dump({"files": self.files, "names": self.names}, f)

    def load(self):
        if os.path.exists(self.persist_path):
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
                self.files = data.get("files", {})
                self.names = data.get("names", {})
        if os.path.exists(self.version_path):
            with open(self.version_path, "r") as f:
                self.version_db = json.load(f)
