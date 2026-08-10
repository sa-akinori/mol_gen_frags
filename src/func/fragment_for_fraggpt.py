import random
import re
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdmolops

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


def assemble_fragments_with_reason(fragments: str) -> tuple[str | None, str]:
    """Rebuild a molecule from FU-SMILES fragments and report why it failed.

    Dummy atoms sharing an isotope label are the two ends of one broken bond, and
    :func:`rdkit.Chem.rdmolops.molzip` rebuilds those bonds. molzip is used rather than
    bonding the neighbors and deleting the dummies by hand because deleting an atom moves
    its neighbor to the end of the neighbor list, which inverts a stereocenter. The **bond
    order of the dummy must agree between the two ends** -- a double bond that was cut is
    written as ``[1*]=C...``, so ends that contradict each other are rejected here; the bond
    itself is made by molzip.

    Args:
        fragments: Fragment SMILES whose attachment points carry paired ``[i*]`` labels.

    Returns:
        Pair ``(smiles, reason)``. On success ``smiles`` is the canonical SMILES and
        ``reason`` is ``"ok"``. On failure ``smiles`` is None and ``reason`` is one of:
            - ``parse_failure``: the fragment list is empty or RDKit could not parse one of
              the fragments.
            - ``dummy_only_fragment``: a fragment holds no heavy atom, i.e. it contributes
              nothing but a bridge between two other fragments.
            - ``unmatched_dummy``: an attachment label is unlabeled, unpaired or used more
              than twice, i.e. a dummy atom would remain in the product.
            - ``invalid_connection``: the two ends of a bond to rebuild are the same atom or
              are bonded already.
            - ``bond_order_mismatch``: the two ends of a bond to rebuild disagree on its bond
              order, i.e. the fragments contradict each other.
            - ``sanitize_failure``: the assembled molecule is not a valid molecule.
            - ``multiple_components``: the fragments do not assemble into a single molecule.
    """
    # Parsed one by one: in one string, the ring-closure digits of two fragments would pair up
    # with each other. The empty fragment the trailing '.' of the prompt makes is not a failure.
    try:
        mols = [Chem.MolFromSmiles(fragment) for fragment in fragments.split('.') if fragment]
    except ValueError:  # RDKit raises instead of returning None for a few malformed SMILES.
        return None, "parse_failure"
    if not mols or any(mol is None for mol in mols):
        return None, "parse_failure"
    # molzip splices a fragment without heavy atoms into a plain bond between its two partners
    # ('[1*][2*].[1*]CC.[2*]OC' -> 'CCOC'), dropping an attachment point the model had to fill.
    if any(mol.GetNumHeavyAtoms() == 0 for mol in mols):
        return None, "dummy_only_fragment"
    combined = mols[0]
    for fragment_mol in mols[1:]:
        combined = Chem.CombineMols(combined, fragment_mol)

    # A broken bond contributes exactly two dummy atoms carrying the same label.
    dummies_by_label: dict[int, list[int]] = defaultdict(list)
    for atom in combined.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dummies_by_label[atom.GetIsotope()].append(atom.GetIdx())
    if any(label == 0 or len(indices) != 2 for label, indices in dummies_by_label.items()):
        return None, "unmatched_dummy"

    molzip_params = rdmolops.MolzipParams()
    molzip_params.label = rdmolops.MolzipLabel.Isotope
    
    pending_bonds: set[frozenset[int]] = set()
    for first_idx, second_idx in dummies_by_label.values():
        first_bond, second_bond = _dummy_bond(combined, first_idx), _dummy_bond(combined, second_idx)
        if first_bond is None or second_bond is None:
            return None, "unmatched_dummy"
        first_neighbor = first_bond.GetOtherAtomIdx(first_idx)
        second_neighbor = second_bond.GetOtherAtomIdx(second_idx)
        neighbors = frozenset((first_neighbor, second_neighbor))
        if (first_neighbor == second_neighbor
                or combined.GetBondBetweenAtoms(first_neighbor, second_neighbor) is not None
                or neighbors in pending_bonds):
            return None, "invalid_connection"
        if first_bond.GetBondType() != second_bond.GetBondType():
            return None, "bond_order_mismatch"
        pending_bonds.add(neighbors)

    try:
        mol = rdmolops.molzip(combined, molzip_params)
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:  # RDKit raises several unrelated exception types here.
        return None, "sanitize_failure"
    if len(Chem.GetMolFrags(mol)) != 1:
        return None, "multiple_components"
    return Chem.MolToSmiles(mol), "ok"


def label_attachment_points(fragments: list[str], rng: random.Random) -> list[str]:
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
    mols = [Chem.MolFromSmiles(fragment) for fragment in fragments]
    unparsed = [fragment for fragment, mol in zip(fragments, mols) if mol is None]
    if unparsed:
        raise ValueError(f"RDKit could not parse the fragments: {unparsed}")
    dummies = [atom for mol in mols for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    labels = list(range(1, len(dummies) + 1))
    rng.shuffle(labels)
    for atom, label in zip(dummies, labels):
        atom.SetIsotope(label)
    return [Chem.MolToSmiles(mol) for mol in mols]


def augment_fusmiles(fusmiles: str, rng: random.Random) -> str:
    """Apply the FragGPT training augmentation to one FU-SMILES fragment set.

    Two invariances are taught: the attachment labels ``1..n`` are relabeled by a random
    permutation (the numbers are arbitrary names of the cut bonds) and the fragments are
    written in random order. The number of sequences is unchanged -- one fragment set stays
    one training sequence -- so the corpus size stays comparable with the other baselines.

    Args:
        fusmiles: Dot-separated fragment SMILES whose attachment points carry paired ``[i*]``
            labels.
        rng: Random generator used for the relabeling and the shuffling.

    Returns:
        The same fragments, relabeled and in random order.
    """
    fragments = [fragment for fragment in fusmiles.split(".") if fragment]
    labels = sorted({int(label) for fragment in fragments for label in re.compile(r"\[(\d+)\*\]").findall(fragment)})
    new_labels = list(range(1, len(labels) + 1))
    rng.shuffle(new_labels)
    relabeling = dict(zip(labels, new_labels))
    augmented = [re.compile(r"\[(\d+)\*\]").sub(lambda m: f"[{relabeling[int(m.group(1))]}*]", fragment) for fragment in fragments]
    rng.shuffle(augmented)
    return ".".join(augmented)
