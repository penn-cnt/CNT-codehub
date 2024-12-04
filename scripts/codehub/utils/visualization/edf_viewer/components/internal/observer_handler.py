from abc import ABC, abstractmethod

class Subject:
    """
    Subject class to allow the BIDS handler to listen for new keywords.
    """
    def add_event_observer(self, observer):
        if observer not in self._event_observers:
            self._event_observers.append(observer)

    def notify_event_observers(self,event):
        for observer in self._event_observers:
            observer.listen_event(self,event)

class Observer(ABC):
    """
    Observer class to allow the BIDS handler to listen for new keywords.

    Args:
        ABC (object): Abstract Base Class object. Enforces the use of abstractmethod to prevent accidental access to listen_keyword without matching
        class in the observer.

    Raises:
        NotImplementedError: Error if the observing class doesn't have the right class object.
    """

    # Listener for BIDS keyword generation to create the correct pathing.
    @abstractmethod
    def listen_event(self):
        raise NotImplementedError("Subclass must implement abstract method")