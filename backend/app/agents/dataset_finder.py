"""
DatasetFinderAgent - Search HuggingFace datasets using OpenRouter AI
Uses OpenRouter for intelligent dataset search and ranking
"""
from typing import List, Dict, Any
import os
import json
import requests
from huggingface_hub import HfApi


class DatasetFinderAgent:
    def __init__(self):
        # Initialize OpenRouter for intelligent dataset search
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        # Get model from env or use default
        self.model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
        
        # Initialize HuggingFace API with token
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            self.hf_api = HfApi(token=hf_token)
            print(f"[DatasetFinder] ✅ HuggingFace API initialized with token")
        else:
            self.hf_api = HfApi()
            print(f"[DatasetFinder] ⚠️ HuggingFace API initialized without token (may have rate limits)")
        
        print(f"[DatasetFinder] ✅ OpenRouter AI enabled (model: {self.model})")

    def find_datasets(self, user_input: str, task_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find relevant datasets from HuggingFace using AI-powered search.
        
        Args:
            user_input: User's prompt or description
            task_type: Type of task (classification, regression, etc.)
            limit: Maximum number of datasets to return
            
        Returns:
            List of dataset dictionaries with id, name, description, url, etc.
        """
        print(f"[DatasetFinder] 🔍 AI-powered search for: {user_input[:100]}...")
        print(f"[DatasetFinder] Task type: {task_type}")
        
        # Use OpenRouter to get dataset search query
        search_query = self._generate_search_query(user_input, task_type)
        print(f"[DatasetFinder] 🔎 Search query: {search_query}")
        
        # Search HuggingFace
        hf_results = self._search_huggingface(search_query, limit=50)
        print(f"[DatasetFinder] 🔍 Found {len(hf_results)} datasets on HuggingFace")
        
        if len(hf_results) == 0:
            print(f"[DatasetFinder] ⚠️ No datasets found - trying broader search...")
        
        # Use AI to rank and select best datasets
        selected = self._ai_rank_datasets(user_input, task_type, hf_results, limit)
        
        # ALWAYS add "Upload CSV" option at the end
        selected.append({
            "id": "upload_csv",
            "name": "📤 Upload Your CSV File",
            "full_name": "custom_upload",
            "description": "Click here to upload your own CSV dataset from your computer",
            "license": "custom",
            "url": None,
            "is_upload_prompt": True,
            "source": "custom_upload"
        })
        
        print(f"[DatasetFinder] ✅ Selected {len(selected)} datasets:")
        for i, ds in enumerate(selected, 1):
            print(f"  {i}. {ds['name']} - {ds['description'][:60]}...")
        
        return selected

    def _generate_search_query(self, user_input: str, task_type: str) -> str:
        """Use AI to generate optimal HuggingFace search query"""
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/AutoML",  # Required by OpenRouter
                    "X-Title": "AutoML Dataset Finder"  # Optional but recommended
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"""Generate a concise HuggingFace dataset search query for this request:
User wants: {user_input}
Task type: {task_type}

Return ONLY 2-5 keywords separated by spaces that would find relevant datasets. No explanation.

Examples:
- For "cats and dogs classification" → "cats dogs image classification"
- For "sentiment analysis" → "sentiment text classification"
- For "house prices" → "housing price regression"
- For "titanic survival" → "titanic classification"

Query:"""
                        }
                    ]
                },
                timeout=10
            )
            
            if response.status_code == 200:
                query = response.json()["choices"][0]["message"]["content"].strip()
                return query
            else:
                print(f"[DatasetFinder] ⚠️ OpenRouter error: {response.status_code}")
                print(f"[DatasetFinder] Response: {response.text}")
                return task_type or "dataset"
                
        except Exception as e:
            print(f"[DatasetFinder] ⚠️ Query generation error: {e}")
            return task_type or "dataset"

    def _search_huggingface(self, query: str, limit: int = 50) -> List[Dict]:
        """Search HuggingFace datasets"""
        try:
            print(f"[DatasetFinder] 🔍 Searching HuggingFace with query: '{query}', limit: {limit}")
            
            # Split query into keywords for better search
            keywords = query.strip().split()
            print(f"[DatasetFinder] 🔑 Keywords: {keywords}")
            
            # Try searching with each keyword combination
            results = []
            dataset_count = 0
            filtered_count = 0
            seen_ids = set()
            
            # First try the full query
            search_queries = [query]
            
            # Also try individual important keywords
            if len(keywords) > 1:
                # Add individual keywords (but not common words)
                common_words = {'and', 'or', 'the', 'a', 'an', 'for', 'with', 'classification', 'regression'}
                search_queries.extend([kw for kw in keywords if kw.lower() not in common_words and len(kw) > 3])
            
            print(f"[DatasetFinder] 🔎 Will try {len(search_queries)} search variations: {search_queries}")
            
            for search_term in search_queries[:3]:  # Limit to 3 searches to avoid too many API calls
                try:
                    print(f"[DatasetFinder]   Searching: '{search_term}'...")
                    search_count = 0
                    
                    for ds in self.hf_api.list_datasets(search=search_term, limit=20):
                        search_count += 1
                        
                        # Skip if already seen
                        if ds.id in seen_ids:
                            continue
                        seen_ids.add(ds.id)
                        
                        dataset_count += 1
                        
                        # Skip private/gated
                        if getattr(ds, "private", False) or getattr(ds, "gated", False):
                            filtered_count += 1
                            continue
                        
                        info = ds.card_data or {}
                        lic_raw = (info.get("license") or "").lower().strip()
                        
                        # Allow most licenses (only reject strictly commercial/proprietary)
                        disallowed = ["proprietary", "other-nc", "cc-by-nc"]
                        if lic_raw and any(d in lic_raw for d in disallowed):
                            filtered_count += 1
                            continue
                        
                        # Get downloads count if available
                        downloads = getattr(ds, "downloads", 0) or 0
                        
                        print(f"[DatasetFinder]     ✅ {ds.id} (downloads={downloads})")
                        
                        results.append({
                            "id": ds.id,
                            "name": ds.id.split("/")[-1].replace("-", " ").title(),
                            "full_name": ds.id,
                            "description": getattr(ds, "description", "No description available") or f"Dataset: {ds.id}",
                            "license": lic_raw or "unspecified",
                            "url": f"https://huggingface.co/datasets/{ds.id}",
                            "tags": getattr(ds, "tags", []) or [],
                            "downloads": downloads
                        })
                        
                        # Stop if we have enough
                        if len(results) >= limit:
                            break
                    
                    print(f"[DatasetFinder]   Found {search_count} for '{search_term}'")
                    
                except Exception as search_e:
                    print(f"[DatasetFinder]   Search failed for '{search_term}': {search_e}")
                
                # Stop if we have enough results
                if len(results) >= limit:
                    break
            
            print(f"[DatasetFinder] 📊 HuggingFace search complete:")
            print(f"[DatasetFinder]   Total found: {dataset_count}")
            print(f"[DatasetFinder]   Filtered out: {filtered_count}")
            print(f"[DatasetFinder]   Returned: {len(results)}")
            
            return results
            
        except Exception as e:
            print(f"[DatasetFinder] ❌ HuggingFace search error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _ai_rank_datasets(self, user_input: str, task_type: str, candidates: List[Dict], limit: int) -> List[Dict]:
        """Use AI to rank datasets by relevance"""
        if not candidates:
            return []
        
        if len(candidates) <= limit:
            return candidates
        
        try:
            # Prepare dataset info for AI
            dataset_info = []
            for ds in candidates[:30]:  # Limit to avoid token overflow
                dataset_info.append({
                    "id": ds["id"],
                    "name": ds["name"],
                    "description": ds["description"][:200],
                    "tags": ds.get("tags", [])[:5]
                })
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/AutoML",  # Required by OpenRouter
                    "X-Title": "AutoML Dataset Finder"  # Optional but recommended
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"""Rank these HuggingFace datasets by relevance for this task:
User wants: {user_input}
Task type: {task_type}

Datasets:
{json.dumps(dataset_info, indent=2)}

Return ONLY a JSON array of the top {limit} dataset IDs in order of relevance.
Format: ["dataset-id-1", "dataset-id-2", ...]
No explanation."""
                        }
                    ]
                },
                timeout=15
            )
            
            if response.status_code == 200:
                ranked_ids = json.loads(response.json()["choices"][0]["message"]["content"].strip())
                
                # Reorder candidates based on AI ranking
                ranked = []
                for ds_id in ranked_ids[:limit]:
                    for ds in candidates:
                        if ds["id"] == ds_id or ds["full_name"] == ds_id:
                            ranked.append(ds)
                            break
                
                # Fill remaining slots with top candidates
                while len(ranked) < limit and len(ranked) < len(candidates):
                    for ds in candidates:
                        if ds not in ranked:
                            ranked.append(ds)
                            break
                
                return ranked[:limit]
            else:
                print(f"[DatasetFinder] ⚠️ Ranking error: {response.status_code}")
                print(f"[DatasetFinder] Response: {response.text}")
                return candidates[:limit]
                
        except Exception as e:
            print(f"[DatasetFinder] ⚠️ AI ranking error: {e}")
            import traceback
            traceback.print_exc()
            return candidates[:limit]