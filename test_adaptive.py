from estimation.backtesting_adapted import compare_strategies, print_comparison
import numpy as np
import pandas as pd

# Synthetic test
np.random.seed(42)
spread = pd.Series(np.random.randn(500).cumsum())

fixed, adaptive = compare_strategies(
    spread=spread,
    theta=0.1,
    mu=0.0,
    sigma=1.0,
    pair_name="Test Pair"
)

print_comparison(fixed, adaptive)