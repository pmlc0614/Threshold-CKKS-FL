"""Pure numeric chunk layout helpers shared by profiles and tensor sparsification."""
import math

def chunk_count(
    total_numel: int,
    chunk_size: int,
) -> int:
    if total_numel <= 0:
        raise ValueError(
            "total_numel must be positive"
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive"
        )

    return math.ceil(
        total_numel / chunk_size
    )


def selected_chunk_count_from_ratio(
    total_numel: int,
    chunk_size: int,
    rho: float,
) -> int:
    """Return ``ceil(rho * d / B)`` capped by ``ceil(d / B)``.

    Here ``d`` is ``total_numel`` and ``B`` is ``chunk_size``. Applying rho
    before the outer chunk-count ceiling keeps plaintext and encrypted profile
    selection identical when the final chunk is only partially populated.
    """

    ratio = float(rho)

    if not math.isfinite(ratio):
        raise ValueError("rho must be finite")

    if not 0.0 < ratio <= 1.0:
        raise ValueError(
            "rho must be greater than 0 and at most 1"
        )

    total_chunks = chunk_count(
        total_numel=total_numel,
        chunk_size=chunk_size,
    )

    return min(
        total_chunks,
        max(
            1,
            math.ceil(ratio * total_numel / chunk_size),
        ),
    )
