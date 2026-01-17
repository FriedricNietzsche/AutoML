"""
Leaderboard Manager Module (Task 5.3 - Phase 2)

Manages model leaderboards with ranking and persistence.
Tracks all training runs and ranks them by performance.

Storage:
    data/projects/{project_id}/leaderboard.json
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from .model_registry import ModelMetadata


# ============================================================================
# Pydantic Models
# ============================================================================

class LeaderboardEntry(BaseModel):
    """Single entry in the leaderboard."""
    rank: int
    run_id: str
    model_name: str
    model_family: str
    metric_name: str
    metric_value: float
    timestamp: str
    hyperparameters: Dict[str, Any]
    training_duration_seconds: float = 0.0


class LeaderboardState(BaseModel):
    """Persistent leaderboard state."""
    project_id: str
    primary_metric: str
    entries: List[LeaderboardEntry]
    last_updated: str


# ============================================================================
# Leaderboard Manager
# ============================================================================

class LeaderboardManager:
    """
    Manages model performance leaderboard.
    
    Features:
    - Add new runs to leaderboard
    - Automatic ranking by metric
    - Persistence to JSON
    - Filter by task type, model family
    - Get top N models
    - Emit events for frontend
    
    Usage:
        leaderboard = LeaderboardManager(base_dir="data/projects")
        
        # Add new run
        leaderboard.add_run(project_id, metadata)
        
        # Get leaderboard
        entries = leaderboard.get_leaderboard(project_id, top_n=10)
        
        # Emit event
        leaderboard.emit_leaderboard_event(emit_fn, entries)
    """
    
    def __init__(self, base_dir: str = "data/projects"):
        """
        Initialize leaderboard manager.
        
        Args:
            base_dir: Base directory for project data
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_leaderboard_path(self, project_id: str) -> Path:
        """Get path to leaderboard.json."""
        project_dir = self.base_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir / "leaderboard.json"
    
    def _load_leaderboard(self, project_id: str) -> LeaderboardState:
        """Load leaderboard from disk."""
        lb_path = self._get_leaderboard_path(project_id)
        
        if not lb_path.exists():
            # Create empty leaderboard
            from datetime import datetime
            return LeaderboardState(
                project_id=project_id,
                primary_metric="f1",  # Default
                entries=[],
                last_updated=datetime.utcnow().isoformat() + "Z"
            )
        
        with open(lb_path, 'r') as f:
            data = json.load(f)
        
        return LeaderboardState(**data)
    
    def _save_leaderboard(self, state: LeaderboardState):
        """Save leaderboard to disk."""
        from datetime import datetime
        
        state.last_updated = datetime.utcnow().isoformat() + "Z"
        
        lb_path = self._get_leaderboard_path(state.project_id)
        
        with open(lb_path, 'w') as f:
            json.dump(state.model_dump(), f, indent=2)
    
    def add_run(
        self,
        project_id: str,
        metadata: ModelMetadata,
        metric_name: Optional[str] = None
    ):
        """
        Add a new run to the leaderboard.
        
        Args:
            project_id: Project identifier
            metadata: ModelMetadata for the run
            metric_name: Metric to rank by (uses primary_metric if None)
        """
        # Load current leaderboard
        state = self._load_leaderboard(project_id)
        
        # Determine metric
        if metric_name is None:
            metric_name = metadata.primary_metric_name
        
        # Update primary metric if needed
        if not state.entries or metric_name != state.primary_metric:
            state.primary_metric = metric_name
        
        # Create entry
        metric_value = metadata.metrics.get(
            metric_name.replace("_", ""),
            metadata.primary_metric_value
        )
        
        entry = LeaderboardEntry(
            rank=0,  # Will be set during ranking
            run_id=metadata.run_id,
            model_name=metadata.model_id,
            model_family=metadata.model_family,
            metric_name=metric_name,
            metric_value=metric_value,
            timestamp=metadata.timestamp,
            hyperparameters=metadata.hyperparameters,
            training_duration_seconds=metadata.training_duration_seconds
        )
        
        # Add to entries
        state.entries.append(entry)
        
        # Re-rank
        self._rerank_entries(state)
        
        # Save
        self._save_leaderboard(state)
    
    def _rerank_entries(self, state: LeaderboardState):
        """Re-rank all entries by metric value (descending)."""
        # Sort by metric value (higher is better)
        state.entries.sort(
            key=lambda e: e.metric_value,
            reverse=True
        )
        
        # Assign ranks
        for i, entry in enumerate(state.entries, start=1):
            entry.rank = i
    
    def get_leaderboard(
        self,
        project_id: str,
        metric_name: Optional[str] = None,
        task_type: Optional[str] = None,
        model_family: Optional[str] = None,
        top_n: Optional[int] = None
    ) -> List[LeaderboardEntry]:
        """
        Get leaderboard entries with optional filtering.
        
        Args:
            project_id: Project identifier
            metric_name: Filter/rank by specific metric (optional)
            task_type: Filter by task type (optional)
            model_family: Filter by model family (optional)
            top_n: Return only top N entries (optional)
            
        Returns:
            List of LeaderboardEntry instances
        """
        state = self._load_leaderboard(project_id)
        
        entries = state.entries
        
        # Re-rank if different metric requested
        if metric_name and metric_name != state.primary_metric:
            # Re-sort by requested metric
            entries = sorted(
                entries,
                key=lambda e: e.metric_value if e.metric_name == metric_name else 0,
                reverse=True
            )
        
        # Apply filters
        if task_type or model_family:
            # Would need to load metadata for each entry
            # For now, skip this optimization
            pass
        
        # Limit to top N
        if top_n:
            entries = entries[:top_n]
        
        return entries
    
    def get_best_run(
        self,
        project_id: str,
        metric_name: Optional[str] = None
    ) -> Optional[LeaderboardEntry]:
        """
        Get the best run from leaderboard.
        
        Args:
            project_id: Project identifier
            metric_name: Metric to optimize (uses primary if None)
            
        Returns:
            Best LeaderboardEntry or None if empty
        """
        entries = self.get_leaderboard(project_id, metric_name=metric_name)
        
        if not entries:
            return None
        
        return entries[0]
    
    def remove_run(self, project_id: str, run_id: str):
        """
        Remove a run from the leaderboard.
        
        Args:
            project_id: Project identifier
            run_id: Run to remove
        """
        state = self._load_leaderboard(project_id)
        
        # Filter out the run
        state.entries = [e for e in state.entries if e.run_id != run_id]
        
        # Re-rank
        self._rerank_entries(state)
        
        # Save
        self._save_leaderboard(state)
    
    def clear_leaderboard(self, project_id: str):
        """
        Clear all entries from leaderboard.
        
        Args:
            project_id: Project identifier
        """
        state = self._load_leaderboard(project_id)
        state.entries = []
        self._save_leaderboard(state)
    
    def emit_leaderboard_event(
        self,
        emit_fn: Callable[[str, Dict], None],
        entries: List[LeaderboardEntry],
        metric_name: str = "f1"
    ):
        """
        Emit LEADERBOARD_UPDATED event for frontend.
        
        Args:
            emit_fn: Event emission function
            entries: Leaderboard entries
            metric_name: Primary metric name
        """
        payload = {
            "metric": metric_name,
            "entries": [
                {
                    "rank": e.rank,
                    "run_id": e.run_id,
                    "model": e.model_family,
                    "metric": e.metric_value,
                    "params": e.hyperparameters,
                    "runtime_s": e.training_duration_seconds
                }
                for e in entries
            ]
        }
        
        emit_fn("LEADERBOARD_UPDATED", payload)
    
    def emit_best_model_event(
        self,
        emit_fn: Callable[[str, Dict], None],
        run_id: str,
        model_id: str,
        metric_name: str,
        metric_value: float
    ):
        """
        Emit BEST_MODEL_UPDATED event.
        
        Args:
            emit_fn: Event emission function
            run_id: Run identifier
            model_id: Model identifier
            metric_name: Metric name
            metric_value: Metric value
        """
        payload = {
            "run_id": run_id,
            "model_id": model_id,
            "metric": {
                "name": metric_name,
                "split": "test",
                "value": metric_value
            }
        }
        
        emit_fn("BEST_MODEL_UPDATED", payload)
    
    def get_statistics(self, project_id: str) -> Dict[str, Any]:
        """
        Get leaderboard statistics.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Statistics dictionary
        """
        entries = self.get_leaderboard(project_id)
        
        if not entries:
            return {
                "total_runs": 0,
                "best_score": None,
                "worst_score": None,
                "avg_score": None,
                "model_families": []
            }
        
        scores = [e.metric_value for e in entries]
        families = list(set(e.model_family for e in entries))
        
        return {
            "total_runs": len(entries),
            "best_score": max(scores),
            "worst_score": min(scores),
            "avg_score": sum(scores) / len(scores),
            "model_families": families,
            "primary_metric": entries[0].metric_name if entries else None
        }
