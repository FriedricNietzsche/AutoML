"""HuggingFace dataset search agent with license validation"""
from huggingface_hub import HfApi
from app.config import Config
from app.agents.license_validator import LicenseValidator
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class DatasetFinderAgent:
    """
    Searches HuggingFace Hub for datasets matching ML task requirements.
    
    Features:
    - Task-aware search (filters by task type)
    - License validation before selection
    - Popularity-based ranking
    - Fallback suggestions when no valid datasets found
    """
    
    def __init__(self):
        """Initialize HF API client and license validator"""
        self.api = HfApi(token=Config.HUGGINGFACEHUB_API_TOKEN)
        self.validator = LicenseValidator()
    
    def find_datasets(
        self,
        task_type: str,
        dataset_hint: Optional[str] = None,
        max_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search for datasets matching the task requirements.
        
        Args:
            task_type: Type of ML task (classification, regression, etc.)
            dataset_hint: Keywords or domain hints for search
            max_results: Maximum number of results (default from config)
            
        Returns:
            List of dataset candidates sorted by:
            1. License validity (valid first)
            2. Downloads (popularity)
            
        Each candidate includes:
            - id: HF dataset ID
            - name: Display name
            - description: Brief description
            - license: License tag
            - license_valid: Whether license is allowed
            - license_reason: Explanation
            - downloads: Download count
            - size: Dataset size in bytes (if available)
        """
        max_results = max_results or Config.MAX_DATASET_SEARCH_RESULTS
        
        results = self._search_hf(task_type, dataset_hint, max_results * 2)
        candidates = self._validate_licenses(results)
        
        # Sort: valid licenses first, then by popularity
        candidates.sort(key=lambda x: (not x["license_valid"], -x["downloads"]))
        
        logger.info(
            f"Found {len(candidates)} candidates, "
            f"{sum(1 for c in candidates if c['license_valid'])} with valid licenses"
        )
        
        return candidates[:max_results]
    
    def _search_hf(
        self,
        task_type: str,
        dataset_hint: Optional[str],
        limit: int
    ) -> List[Any]:
        """
        Query HuggingFace Hub API for datasets.
        
        Args:
            task_type: ML task type
            dataset_hint: Search keywords
            limit: Max results from API
            
        Returns:
            List of dataset info objects from HF API
        """
        # Build search query
        search_terms = []
        
        if dataset_hint:
            search_terms.append(dataset_hint)
        
        # Add task-specific keywords
        task_keywords = {
            "classification": ["classification", "labeled", "categories"],
            "vision": ["image", "vision", "visual", "photo"],
            "nlp": ["text", "language", "nlp", "corpus"],
            "regression": ["regression", "continuous", "prediction"],
            "time_series": ["time series", "temporal", "sequential"],
        }
        
        if task_type in task_keywords:
            search_terms.extend(task_keywords[task_type])
        
        query = " ".join(search_terms) if search_terms else None
        
        logger.info(f"Searching HF for: {query} (task={task_type})")
        
        try:
            # Search datasets with filters
            datasets = list(self.api.list_datasets(
                search=query,
                limit=limit,
                sort="downloads",
                direction=-1,  # Descending (most popular first)
            ))
            
            logger.info(f"HF API returned {len(datasets)} datasets")
            return datasets
            
        except Exception as e:
            logger.error(f"HF API search failed: {e}")
            return []
    
    def _validate_licenses(self, datasets: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract metadata and validate licenses for each dataset.
        
        Args:
            datasets: Raw dataset objects from HF API
            
        Returns:
            List of enriched candidate dicts with license validation
        """
        candidates = []
        
        for ds in datasets:
            try:
                # Extract license info
                card_data = getattr(ds, 'cardData', None) or {}
                license_tag = card_data.get("license")
                
                # Validate license
                is_valid, reason = self.validator.is_allowed(license_tag)
                
                # Build candidate dict
                candidate = {
                    "id": ds.id,
                    "name": ds.id.split("/")[-1] if "/" in ds.id else ds.id,
                    "description": (card_data.get("task_categories", []) or [""])[:200] if card_data else "",
                    "license": license_tag,
                    "license_valid": is_valid,
                    "license_reason": reason,
                    "downloads": getattr(ds, 'downloads', 0) or 0,
                    "size": None,  # Size extraction can be added if needed
                }
                
                candidates.append(candidate)
                
            except Exception as e:
                logger.warning(f"Failed to process dataset {getattr(ds, 'id', '?')}: {e}")
                continue
        
        return candidates
    
    def get_best_dataset(
        self,
        task_type: str,
        dataset_hint: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find the single best dataset for the task.
        
        Returns the most popular dataset with a valid license,
        or None if no valid datasets exist.
        
        Args:
            task_type: ML task type
            dataset_hint: Search keywords
            
        Returns:
            Best dataset candidate or None
        """
        candidates = self.find_datasets(task_type, dataset_hint, max_results=10)
        
        # Filter to only valid licenses
        valid = [c for c in candidates if c["license_valid"]]
        
        if not valid:
            logger.warning("No datasets with valid licenses found")
            return None
        
        best = valid[0]
        logger.info(f"Selected best dataset: {best['id']} (license={best['license']})")
        return best