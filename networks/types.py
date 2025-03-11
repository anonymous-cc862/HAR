
import flax
import collections
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

Batch = collections.namedtuple(
    'Batch',
    ['X', 'num_idx', 'cate_idx', 'y'])

MetaBatch = collections.namedtuple(
    'MetaBatch',
    ['X', 'X_tar', 'X_support'])

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
InfoDict = Dict[str, Any]

