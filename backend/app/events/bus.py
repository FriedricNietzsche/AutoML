from typing import Callable, Dict, Any

class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, Callable[[Dict[str, Any]], None]] = {}

    def subscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    def publish(self, event_type: str, event_data: Dict[str, Any]) -> None:
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                callback(event_data)

    def unsubscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(callback)
            if not self.subscribers[event_type]:
                del self.subscribers[event_type]