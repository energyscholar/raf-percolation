"""
Binary polymer model and RAF detection.

Implements the Kauffman (1986) / Hordijk & Steel (2004) binary polymer
model with random catalysis, and the polynomial-time RAF detection
algorithm.

References:
    Kauffman, J. Theor. Biol. 119:1-24 (1986)
    Hordijk & Steel, J. Theor. Biol. 227:451-461 (2004)
    Hordijk, Steel & Kauffman, Int. J. Mol. Sci. 12:3085-3101 (2011)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BinaryPolymerCRS:
    """A catalytic reaction system on binary polymers.

    Attributes:
        n: Maximum string length.
        molecules: List of all bit strings (length 1..n).
        food: Indices of food molecules (length 1 and 2).
        reactions: List of (reactant_indices, product_indices) tuples.
        catalysis: Boolean array of shape (n_molecules, n_reactions).
    """
    n: int
    molecules: list
    food: np.ndarray
    reactions: list
    catalysis: np.ndarray

    @property
    def n_molecules(self):
        return len(self.molecules)

    @property
    def n_reactions(self):
        return len(self.reactions)


def generate_molecules(n: int) -> list:
    """Generate all binary strings of length 1 through n."""
    molecules = []
    for length in range(1, n + 1):
        for val in range(2**length):
            molecules.append(format(val, f'0{length}b'))
    return molecules


def generate_reactions(molecules: list) -> list:
    """Generate all ligation/cleavage reactions.

    Following the Hordijk & Steel convention: each reaction is a
    reversible ligation-cleavage pair, counted ONCE.  A reaction
    r has reactants = {left, right} (the two fragments) and
    products = {joined} (the concatenated string), with the
    understanding that the reverse (cleavage) is implicit.

    This matches the standard |R| = (n-2)*2^{n+1} + 4 formula.

    Returns:
        List of (frozenset of reactant indices, frozenset of product indices).
        For RAF purposes, both directions generate available molecules:
        if all reactants are available, products become available, AND
        if the product is available, reactants become available.
    """
    mol_to_idx = {m: i for i, m in enumerate(molecules)}
    reactions = []

    for mol in molecules:
        if len(mol) < 2:
            continue
        # Split mol at each internal position
        for pos in range(1, len(mol)):
            left = mol[:pos]
            right = mol[pos:]
            if left in mol_to_idx and right in mol_to_idx:
                mol_idx = mol_to_idx[mol]
                left_idx = mol_to_idx[left]
                right_idx = mol_to_idx[right]
                # Single reaction: {left, right} <-> {mol}
                reactions.append((
                    frozenset([left_idx, right_idx]),
                    frozenset([mol_idx])
                ))

    # Deduplicate
    seen = set()
    unique = []
    for reactants, products in reactions:
        key = (reactants, products)
        if key not in seen:
            seen.add(key)
            unique.append((reactants, products))

    return unique


def build_crs(n: int, p: float, rng: Optional[np.random.Generator] = None) -> BinaryPolymerCRS:
    """Build a binary polymer CRS with random catalysis.

    Args:
        n: Maximum string length.
        p: Catalysis probability per (molecule, reaction) pair.
        rng: Random number generator (for reproducibility).

    Returns:
        BinaryPolymerCRS instance.
    """
    if rng is None:
        rng = np.random.default_rng()

    molecules = generate_molecules(n)
    reactions = generate_reactions(molecules)

    n_mol = len(molecules)
    n_rxn = len(reactions)

    # Random catalysis assignment
    catalysis = rng.random((n_mol, n_rxn)) < p

    # Food set: monomers and dimers (length 1 and 2)
    food = np.array([
        i for i, m in enumerate(molecules) if len(m) <= 2
    ])

    return BinaryPolymerCRS(
        n=n, molecules=molecules, food=food,
        reactions=reactions, catalysis=catalysis
    )


def detect_raf(crs: BinaryPolymerCRS) -> Optional[np.ndarray]:
    """Detect the maximal RAF set using the Hordijk-Steel algorithm.

    Iteratively removes reactions that are not catalyzed by any
    molecule in the closure of the food set, until convergence.

    Args:
        crs: A BinaryPolymerCRS instance.

    Returns:
        Boolean array over reactions (True = in maxRAF), or None if
        no RAF exists.
    """
    n_mol = crs.n_molecules
    n_rxn = crs.n_reactions

    # Start with all reactions active
    active_rxn = np.ones(n_rxn, dtype=bool)

    for _ in range(n_rxn):  # Max iterations = number of reactions
        # Compute closure of food set under active reactions
        available = np.zeros(n_mol, dtype=bool)
        available[crs.food] = True

        changed = True
        while changed:
            changed = False
            for r_idx in range(n_rxn):
                if not active_rxn[r_idx]:
                    continue
                reactants, products = crs.reactions[r_idx]
                # Check if reaction is catalyzed by an available molecule
                catalyzed = np.any(crs.catalysis[:, r_idx] & available)
                if not catalyzed:
                    continue
                # Forward: all reactants available -> products available
                if all(available[ri] for ri in reactants):
                    for pi in products:
                        if not available[pi]:
                            available[pi] = True
                            changed = True
                # Reverse (cleavage): all products available -> reactants available
                if all(available[pi] for pi in products):
                    for ri in reactants:
                        if not available[ri]:
                            available[ri] = True
                            changed = True

        # Remove reactions that are not supported
        new_active = np.zeros(n_rxn, dtype=bool)
        for r_idx in range(n_rxn):
            if not active_rxn[r_idx]:
                continue
            reactants, products = crs.reactions[r_idx]
            catalyzed = np.any(crs.catalysis[:, r_idx] & available)
            # Reaction stays if catalyzed AND operable in at least
            # one direction (all reactants OR all products available)
            reactants_ok = all(available[ri] for ri in reactants)
            products_ok = all(available[pi] for pi in products)
            if catalyzed and (reactants_ok or products_ok):
                new_active[r_idx] = True

        if np.array_equal(new_active, active_rxn):
            break
        active_rxn = new_active

    if not np.any(active_rxn):
        return None
    return active_rxn


def raf_size(crs: BinaryPolymerCRS) -> int:
    """Return the size of the maximal RAF (0 if none exists)."""
    result = detect_raf(crs)
    if result is None:
        return 0
    return int(np.sum(result))
