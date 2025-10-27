
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..core.metrics import RequestMetrics

class BaseStorage(ABC):
    """Abstract base class for metrics storage backends"""
    @abstractmethod
    def store_metrics(self, metrics: RequestMetrics) -> bool:
        """
        Store request metrics
        Args:
            metrics: RequestMetrics object to store
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    def get_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[RequestMetrics]:
        
        raise NotImplementedError()

    @abstractmethod
    def get_aggregated_metrics(
        self,
        time_window_minutes: int = 60
    ) -> Dict[str, Any]:
     
        raise NotImplementedError()

    @abstractmethod
    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """
        Clean up old metrics data
        Args:
            days_to_keep: Number of days to keep
        Returns:
            Number of records deleted
        """
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        """Close storage connections"""
        raise NotImplementedError()

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if storage is healthy
        Returns:
            True if healthy, False otherwise
        """
        raise NotImplementedError()
