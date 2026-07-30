"""FU-SMILES helpers for the FragGPT baseline.

FragGPT (Yue et al., *Chem. Sci.* 2024) represents a molecule as its BRICS fragments whose
cut bonds carry **paired** dummy labels, e.g. ``[1*]CCO.[1*]c1ccccc1``: the two ``[1*]``
atoms mark the two ends of one broken bond. That string is called FU-SMILES here. This
project writes the fragments with ``.`` as the separator instead of the ``<sep>`` token of
the original implementation, so a FU-SMILES string stays a plain (multi-component) SMILES
and the ``entropy/gpt2_zinc_87m`` vocabulary can be reused unchanged.

The module covers the three operations FragGPT needs:

    - :func:`split_fragments` / :func:`assemble_fragments`: FU-SMILES <-> molecule.
    - :func:`renumber_open_attachments`: label the bare ``*`` of a prompted fragment set,
      which is what turns a fragment set into an inference prompt.
    - :func:`augment_fusmiles`: the training-time augmentation of the original work
      (random relabeling of the attachment points + random fragment order).

Reassembly keeps the bond order of the broken bond (a dummy attached by ``=`` rebuilds a
double bond) but cannot restore the E/Z configuration of a cut double bond, so the
assembled molecule may lose stereochemistry -- the same limitation SAFE has in
``src/gen_frags/safe_frags.py``.
"""

import random
from collections import defaultdict

from rdkit import Chem

# Fragments are separated by ".", i.e. a FU-SMILES string is a multi-component SMILES.
FRAGMENT_SEPARATOR = "."
# RDKit atomic number of a dummy atom ("*"); its isotope holds the attachment-point label.
DUMMY_ATOMIC_NUM = 0
# Reason returned by assemble_fragments_with_reason on success.
ASSEMBLY_OK = "ok"


def split_fragments(fusmiles: str) -> list[str]:
    """Split a FU-SMILES string into its fragments.

    Args:
        fusmiles: Dot-separated fragment SMILES, e.g. ``[1*]CCO.[1*]c1ccccc1``.

    Returns:
        The fragment SMILES in the order they appear. Empty pieces (leading/trailing or
        repeated separators, which a language model can emit) are dropped.
    """
    return [fragment for fragment in fusmiles.split(FRAGMENT_SEPARATOR) if fragment]


def parse_fragments(fragments: list[str]) -> list[Chem.Mol]:
    """Parse fragment SMILES with RDKit.

    Args:
        fragments: Fragment SMILES, as returned by :func:`split_fragments`.

    Returns:
        One molecule per fragment, in input order.

    Raises:
        ValueError: If RDKit cannot parse one of the fragments.
    """
    mols = []
    for fragment in fragments:
        mol = Chem.MolFromSmiles(fragment)
        if mol is None:
            raise ValueError(f"RDKit could not parse the fragment: {fragment!r}")
        mols.append(mol)
    return mols


def _dummy_bond(mol: Chem.Mol, dummy_idx: int) -> Chem.Bond | None:
    """Return the single bond a dummy atom is attached by.

    Args:
        mol: Molecule holding the dummy atom.
        dummy_idx: Atom index of the dummy atom.

    Returns:
        The bond, or None when the dummy atom does not have exactly one bond (a lone ``*``
        or a ``*`` with several bonds cannot mark one broken bond).
    """
    bonds = mol.GetAtomWithIdx(dummy_idx).GetBonds()
    return bonds[0] if len(bonds) == 1 else None


def assemble_fragments_with_reason(fragments: list[str]) -> tuple[str | None, str]:
    """Rebuild a molecule from FU-SMILES fragments and report why it failed.

    Dummy atoms sharing an isotope label are the two ends of one broken bond: the atoms they
    are attached to are bonded again and the dummies are deleted. The **bond order of the
    dummy is reproduced** -- of the two dummy bonds the one that is not single wins -- because
    a double bond that was cut is written as ``[1*]=C...`` and rebuilding it as a single bond
    would silently change the molecule.

    Args:
        fragments: Fragment SMILES whose attachment points carry paired ``[i*]`` labels.

    Returns:
        Pair ``(smiles, reason)``. On success ``smiles`` is the canonical SMILES of the
        assembled molecule and ``reason`` is :data:`ASSEMBLY_OK`. On failure ``smiles`` is
        None and ``reason`` is one of:
            - ``parse_failure``: the fragment list is empty or RDKit could not parse one of
              the fragments.
            - ``unmatched_dummy``: an attachment label is unlabeled, unpaired or used more
              than twice, i.e. a dummy atom would remain in the product.
            - ``invalid_connection``: the two ends of a bond to rebuild are the same atom or
              are bonded already.
            - ``sanitize_failure``: the assembled molecule is not a valid molecule.
            - ``multiple_components``: the fragments do not assemble into a single molecule.
    """
    # The fragments are parsed one by one and combined afterwards: joining them into one
    # string first would let the ring-closure digits of two fragments pair up with each other.
    try:
        mols = parse_fragments(fragments)
    except ValueError:
        return None, "parse_failure"
    if not mols:
        return None, "parse_failure"
    combined = mols[0]
    for fragment_mol in mols[1:]:
        combined = Chem.CombineMols(combined, fragment_mol)

    # Group the dummy atoms by their attachment label; a broken bond contributes two of them.
    dummies_by_label: dict[int, list[int]] = defaultdict(list)
    for atom in combined.GetAtoms():
        if atom.GetAtomicNum() == DUMMY_ATOMIC_NUM:
            dummies_by_label[atom.GetIsotope()].append(atom.GetIdx())
    if any(label == 0 or len(indices) != 2 for label, indices in dummies_by_label.items()):
        return None, "unmatched_dummy"

    rwmol = Chem.RWMol(combined)
    for first_idx, second_idx in dummies_by_label.values():
        first_bond, second_bond = _dummy_bond(rwmol, first_idx), _dummy_bond(rwmol, second_idx)
        if first_bond is None or second_bond is None:
            return None, "unmatched_dummy"
        first_neighbor = first_bond.GetOtherAtomIdx(first_idx)
        second_neighbor = second_bond.GetOtherAtomIdx(second_idx)
        if first_neighbor == second_neighbor or rwmol.GetBondBetweenAtoms(first_neighbor, second_neighbor) is not None:
            return None, "invalid_connection"
        bond_types = (first_bond.GetBondType(), second_bond.GetBondType())
        # The cut bond order is carried by whichever dummy bond is not single.
        bond_type = next((bt for bt in bond_types if bt != Chem.BondType.SINGLE), Chem.BondType.SINGLE)
        rwmol.AddBond(first_neighbor, second_neighbor, bond_type)

    # Removing an atom shifts every higher index, so delete from the back.
    for dummy_idx in sorted((idx for indices in dummies_by_label.values() for idx in indices), reverse=True):
        rwmol.RemoveAtom(dummy_idx)

    mol = rwmol.GetMol()
    try:
        Chem.SanitizeMol(mol)
        # Deleting the dummies invalidates the stereo perceived when the fragments were read.
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:  # RDKit raises several unrelated exception types here.
        return None, "sanitize_failure"
    if len(Chem.GetMolFrags(mol)) != 1:
        return None, "multiple_components"
    return Chem.MolToSmiles(mol), ASSEMBLY_OK


def assemble_fragments(fragments: list[str]) -> str | None:
    """Rebuild a molecule from FU-SMILES fragments.

    Thin wrapper around :func:`assemble_fragments_with_reason` for callers that do not need
    the failure reason.

    Args:
        fragments: Fragment SMILES whose attachment points carry paired ``[i*]`` labels.

    Returns:
        Canonical SMILES of the single assembled molecule, or None when the fragments do not
        assemble into one dummy-free molecule.
    """
    return assemble_fragments_with_reason(fragments)[0]


def renumber_open_attachments(fragments: list[str], rng: random.Random) -> list[str]:
    """Give every open attachment point of a prompted fragment set its own new label.

    This is the inference-side step of FragGPT: the prompted fragments come from the shared
    test split (``pass_fragments``), where the attachment points are bare ``*`` because the
    partner of each cut was discarded when the fragment subset was sampled. Every ``*`` gets
    a distinct label of ``1..k``, so no label is paired inside the prompt and the model has
    to write the partner fragment for each of them. The labels are dealt out in random order
    because training saw arbitrary label assignments (see :func:`augment_fusmiles`).

    Args:
        fragments: Fragment SMILES carrying unlabeled ``*`` attachment points.
        rng: Random generator used to shuffle the ``1..k`` labels.

    Returns:
        The fragments in input order, rewritten with ``[i*]`` attachment points.

    Raises:
        ValueError: If RDKit cannot parse one of the fragments.
    """
    mols = parse_fragments(fragments)
    dummies = [atom for mol in mols for atom in mol.GetAtoms() if atom.GetAtomicNum() == DUMMY_ATOMIC_NUM]
    labels = list(range(1, len(dummies) + 1))
    rng.shuffle(labels)
    for atom, label in zip(dummies, labels):
        atom.SetIsotope(label)
    return [Chem.MolToSmiles(mol) for mol in mols]


def augment_fusmiles(fragments: list[str], rng: random.Random) -> list[str]:
    """Apply the FragGPT training augmentation to one FU-SMILES fragment set.

    Two invariances are taught: the attachment labels ``1..n`` are relabeled by a random
    permutation (the numbers are arbitrary names of the cut bonds) and the fragments are
    written in random order. The number of sequences is unchanged -- one fragment set stays
    one training sequence -- so the corpus size stays comparable with the other baselines.

    Args:
        fragments: Fragment SMILES whose attachment points carry paired ``[i*]`` labels.
        rng: Random generator used for the relabeling and the shuffling.

    Returns:
        The same fragments, relabeled and in random order.

    Raises:
        ValueError: If RDKit cannot parse one of the fragments.
    """
    mols = parse_fragments(fragments)
    labels = sorted({atom.GetIsotope() for mol in mols for atom in mol.GetAtoms()
                     if atom.GetAtomicNum() == DUMMY_ATOMIC_NUM and atom.GetIsotope() != 0})
    new_labels = list(range(1, len(labels) + 1))
    rng.shuffle(new_labels)
    relabeling = dict(zip(labels, new_labels))
    for mol in mols:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == DUMMY_ATOMIC_NUM and atom.GetIsotope() != 0:
                atom.SetIsotope(relabeling[atom.GetIsotope()])
    augmented = [Chem.MolToSmiles(mol) for mol in mols]
    rng.shuffle(augmented)
    return augmented
