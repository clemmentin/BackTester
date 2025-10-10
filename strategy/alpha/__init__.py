from strategy.alpha.alpha_engine import AlphaEngine, AlphaSource
from strategy.alpha.market_detector import (
    MarketDetector,
    MarketState,
    MarketRegime,
    MacroRegime,
)
from strategy.alpha.alpha_normalization import AlphaNormalizer
from strategy.alpha.momentum_alpha import MomentumAlphaModule
from strategy.alpha.reversal_alpha import ReversalAlphaModule
from strategy.alpha.price_alpha import PriceAlphaModule

# Legacy modules (deprecated but kept for backward compatibility)
# from strategy.alpha.value_alpha import ValueAlphaModule  # DEPRECATED
# from strategy.alpha.quality_alpha import QualityAlphaModule  # DEPRECATED

__all__ = [
    # Core
    "AlphaEngine",
    "AlphaSource",
    "MarketDetector",
    "MarketState",
    "MarketRegime",
    "MacroRegime",
    "AlphaNormalizer",
    # Three-factor modules
    "MomentumAlphaModule",
    "ReversalAlphaModule",
    "PriceAlphaModule",
    # Deprecated (commented out)
    # 'ValueAlphaModule',
    # 'QualityAlphaModule',
]
