from abc import ABC, abstractmethod
from ..point import Point


''' Classe Abstraite Cell'''

class Cell(ABC):
    CELL_TYPE = "Generic"
    _global_counter = 0
    _counter = 0

    def __init__(self, ids: int = None, pos: Point = None):
        if type(self) is Cell:
            raise TypeError("Cell is abstract and cannot be instantiated directly. Use a subclass instead.")

        if ids is not None and not isinstance(ids, int):
            raise TypeError(f"ids must be an int or None, got {type(ids).__name__}")

        cls = self.__class__
        if ids is None:
            cls._counter += 1
            Cell._global_counter += 1
            self._ids = cls._counter
            self._global_id = Cell._global_counter
        else:
            self._ids = ids
            self._global_id = ids

        self._cell_type = self.CELL_TYPE

        if pos is not None and not isinstance(pos, Point):
            raise TypeError(f"pos must be a Point or None, got {type(pos).__name__}")
        self.pos = pos.coord if pos else None

        # Historique des positions
        self._history = [tuple(self.pos)]

    @abstractmethod
    def activate(self):
        pass

    def move_to(self, new_pos: Point):
        """Déplacement contrôlé selon le type de cellule."""
        if not isinstance(new_pos, Point):
            raise TypeError(f"new_pos must be a Point, got {type(new_pos).__name__}")
        
        # Seules les Microglia peuvent se déplacer
        if isinstance(self, Microglia):
            self.pos = new_pos.coord
            self._history.append(tuple(self.pos))
        else:
            raise AttributeError(f"{self.__class__.__name__} cells cannot move!")

    def __str__(self):
        return (f"{self.__class__.__name__}(ids={self._ids}, "
                f"global_id={self._global_id}, "
                f"cell_type={self._cell_type}, pos={self.pos}, "
                f"history={self._history})")

    def __repr__(self):
        return self.__str__()


# ---- Sous-classes ----

class Neuron(Cell):
    CELL_TYPE = "Neuron"
    _counter = 0

    def activate(self):
        print(f"Neuron {self._ids} fires!")


class Astrocyte(Cell):
    CELL_TYPE = "Astrocyte"
    _counter = 0

    def activate(self):
        print(f"Astrocyte {self._ids} regulates neurotransmitters!")


class Microglia(Cell):
    CELL_TYPE = "Microglia"
    _counter = 0

    def activate(self):
        print(f"Microglia {self._ids} removes debris!")
