class AssetStore:
    def upload_asset(self, asset_id: str, asset_data: bytes) -> None:
        """Uploads an asset to the store."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def retrieve_asset(self, asset_id: str) -> bytes:
        """Retrieves an asset from the store."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def delete_asset(self, asset_id: str) -> None:
        """Deletes an asset from the store."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def list_assets(self) -> list:
        """Lists all assets in the store."""
        raise NotImplementedError("This method should be implemented by subclasses.")