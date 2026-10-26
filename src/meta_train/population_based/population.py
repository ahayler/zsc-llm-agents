from dataclasses import dataclass, field
from typing import List, Optional, Union
from pathlib import Path
import pickle

import numpy as np

from utils.saving import CodeXPExperimentResults


@dataclass
class ScoreEntry:
    """A single evaluation score for a genotype."""
    experiment_results: CodeXPExperimentResults
    score: float  # Mean XP score (ignoring NaNs)


@dataclass
class Genotype:
    """A single candidate CFD in the population."""
    cfd: str
    step_created_at: int
    score_history: List[ScoreEntry] = field(default_factory=list)
    step_died_at: Optional[int] = None
    parents: List["Genotype"] = field(default_factory=list)
    mutation_operator_name: Optional[str] = None


@dataclass
class InitializationHistoryEntry:
    """Records the initial population seeding."""
    initial_genotypes: List[Genotype]
    step: int


@dataclass
class BinaryTournamentHistoryEntry:
    """Records a binary tournament step."""
    evaluated_genotypes: List[Genotype]
    scores: List[float]
    new_genotype: Genotype
    mutation_operator_name: str
    step: int


@dataclass
class TournamentHistoryEntry:
    """Records a tournament step (supports any tournament size)."""
    evaluated_genotypes: List[Genotype]
    scores: List[float]
    parent_genotype: Genotype
    new_genotype: Genotype
    eliminated_genotype: Genotype
    mutation_operator_name: str
    step: int


HistoryEntry = Union[InitializationHistoryEntry, BinaryTournamentHistoryEntry, TournamentHistoryEntry]


class Population:
    """
    Manages a population of genotypes for population-based CFD optimization.
    
    Keeps track of both alive and dead genotypes, maintains a step counter,
    and records history of population changes.
    """
    
    def __init__(self):
        self._genotypes: List[Genotype] = []
        self._step_counter: int = 0
        self.history: List[HistoryEntry] = []
    
    @property
    def current_step(self) -> int:
        """Return the current step counter value."""
        return self._step_counter
    
    def add_genotype(self, genotype: Genotype) -> None:
        """Add a genotype to the population and increment the step counter."""
        self._genotypes.append(genotype)
        self._step_counter += 1
    
    def remove_genotype(self, genotype: Genotype) -> None:
        """Mark a genotype as dead and increment the step counter."""
        if genotype not in self._genotypes:
            raise ValueError("Genotype not found in population")
        if genotype.step_died_at is not None:
            raise ValueError("Genotype is already dead")
        genotype.step_died_at = self._step_counter
        self._step_counter += 1
    
    def get_alive(self) -> List[Genotype]:
        """Return all genotypes that are still alive (step_died_at is None)."""
        return [g for g in self._genotypes if g.step_died_at is None]
    
    def get_dead(self) -> List[Genotype]:
        """Return all genotypes that have been removed (step_died_at is not None)."""
        return [g for g in self._genotypes if g.step_died_at is not None]
    
    def sample_alive(self, n: int, rng: Optional[np.random.Generator] = None) -> List[Genotype]:
        """
        Sample n alive genotypes without replacement.
        
        Args:
            n: Number of genotypes to sample.
            rng: Optional numpy random generator for reproducibility.
        
        Returns:
            List of sampled genotypes.
        
        Raises:
            ValueError: If n is greater than the number of alive genotypes.
        """
        alive = self.get_alive()
        if n > len(alive):
            raise ValueError(f"Cannot sample {n} genotypes from {len(alive)} alive genotypes")
        
        if rng is None:
            rng = np.random.default_rng()
        
        indices = rng.choice(len(alive), size=n, replace=False)
        return [alive[i] for i in indices]
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the population to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'genotypes': self._genotypes,
                'step_counter': self._step_counter,
                'history': self.history,
            }, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Population":
        """Load a population from a pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        population = cls()
        population._genotypes = data['genotypes']
        population._step_counter = data['step_counter']
        population.history = data['history']
        return population

