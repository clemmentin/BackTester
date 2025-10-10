from dataclasses import dataclass, field
from typing import Dict, Set, Optional


@dataclass
class RawAlphaSignal:
    symbol: str
    score: float
    confidence: float
    components: Optional[Dict[str, float]] = None


@dataclass
class RiskBudget:
    target_portfolio_exposure: float
    max_position_count: int


@dataclass
class StratifiedCandidatePool:
    conservative_candidates: Set[str]
    aggressive_candidates: Set[str]
    allocation_ratio: Dict[str, float]

    def get_all_candidates(self) -> Set[str]:
        return self.conservative_candidates | self.aggressive_candidates

    @property
    def all_candidates(self) -> Set[str]:
        return self.conservative_candidates.union(self.aggressive_candidates)

    @property
    def conservative_only(self) -> Set[str]:
        return self.conservative_candidates - self.aggressive_candidates

    @property
    def aggressive_only(self) -> Set[str]:
        return self.aggressive_candidates - self.conservative_candidates

    @property
    def consensus_candidates(self) -> Set[str]:
        return self.conservative_candidates.intersection(self.aggressive_candidates)


# Type aliases
RawAlphaSignalDict = Dict[str, RawAlphaSignal]
CandidatePool = Set[str]
FinalTargetPortfolio = Dict[str, float]
