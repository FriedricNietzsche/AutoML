class LocalDiskStore:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def save_asset(self, asset_id: str, data: bytes) -> str:
        file_path = f"{self.base_path}/{asset_id}"
        with open(file_path, 'wb') as file:
            file.write(data)
        return file_path

    def load_asset(self, asset_id: str) -> bytes:
        file_path = f"{self.base_path}/{asset_id}"
        with open(file_path, 'rb') as file:
            return file.read()

    def delete_asset(self, asset_id: str) -> None:
        file_path = f"{self.base_path}/{asset_id}"
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass

    def asset_exists(self, asset_id: str) -> bool:
        file_path = f"{self.base_path}/{asset_id}"
        return os.path.exists(file_path)