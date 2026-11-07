from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from .DeepDDI import ROOT

def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError('Invalid or empty SMILES')
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

FEATURE_PROTOCOL = 'deepddix-explicit-h-morgan-count-r2-v1'
IDENTITY_PROTOCOL = 'deepddix-reference-application-identities-v1'


def atom_invariants(mol):
    table = Chem.GetPeriodicTable()
    values = []
    for atom in mol.GetAtoms():
        components = [atom.GetAtomicNum(), atom.GetTotalDegree(),
                      atom.GetTotalNumHs(includeNeighbors=False), atom.GetFormalCharge(),
                      int(atom.GetMass() - table.GetAtomicWeight(atom.GetAtomicNum()))]
        if atom.IsInRing():
            components.append(1)
        value = 0
        for component in components:
            value = (value ^ ((int(component) & 0xffffffff) + 0x9e3779b9 +
                              ((value << 6) & 0xffffffff) + (value >> 2))) & 0xffffffff
        values.append(value)
    return values


def fingerprint(mol):
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError('A nonempty molecule is required')
    expanded = Chem.AddHs(mol)
    return AllChem.GetMorganFingerprint(
        expanded, 2, invariants=atom_invariants(expanded), useChirality=False,
        useBondTypes=True, useFeatures=False, useCounts=True,
        includeRedundantEnvironments=False)


def load_reference_molecule(path: str | Path):
    path = str(path)
    with rdkit_log_block():
        mol = Chem.MolFromMolFile(path)
        if mol is None:
            mol = Chem.MolFromMolFile(path, sanitize=False, removeHs=False)
            if mol is None:
                raise ValueError(f'Cannot parse reference structure: {path}')
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)
    return mol

class rdkit_log_block:

    def __enter__(self):
        from rdkit import rdBase
        self.block = rdBase.BlockLogs()
        return self

    def __exit__(self, *args):
        del self.block

class DrugResolver:

    def __init__(self, root: str | Path=ROOT):
        self.root = Path(root)
        self.features = pd.read_csv(self.root / 'data/drug_tanimoto_PCA50.csv', index_col=0)
        if not self.features.index.is_unique or self.features.shape[1] != 50:
            raise ValueError('Invalid frozen PCA50 table')
        table = pd.read_csv(self.root / 'data/drug_identity.tsv', sep='\t', dtype=str).fillna('')
        self.names, self.structures, self.id_structures = ({}, {}, {})
        self.known_ids = set(table.drug_id) | set(self.features.index)
        for row in table.itertuples(index=False):
            for key in [row.drug_id, row.name]:
                if key:
                    self.names.setdefault(key.casefold(), set()).add(row.drug_id)
            if row.canonical_smiles:
                self.structures.setdefault(row.canonical_smiles, set()).add(row.drug_id)
                self.id_structures.setdefault(row.drug_id, set()).add(row.canonical_smiles)
        self.reference_fps = None
        self.feature_validation = None
        z = np.load(self.root / 'data/PCA_tanimoto_model_50.npz', allow_pickle=False)
        self.components, self.mean = (z['components'], z['mean'])
        self.reference_ids = z['reference_ids'].tolist()
        if self.components.shape != (50, len(self.reference_ids)) or bool(z['whiten']):
            raise ValueError('Invalid PCA transform')

    def resolve(self, name: str, smiles: str='', explicit_id: str='') -> tuple[str, str]:
        name, smiles, explicit_id = (name.strip(), smiles.strip(), explicit_id.strip().upper())
        canon = canonical_smiles(smiles) if smiles else ''
        if explicit_id:
            if explicit_id not in self.known_ids:
                raise ValueError(f'Unknown explicit DrugBank ID: {explicit_id}')
            candidates = {explicit_id}
        else:
            candidates = self.names.get(name.casefold(), set())
        if candidates:
            if len(candidates) > 1:
                if canon:
                    candidates = candidates & self.structures.get(canon, set())
                if len(candidates) != 1:
                    raise ValueError(f'Ambiguous name; supply an explicit DrugBank ID: {name}')
            drug_id = next(iter(candidates))
            allowed = self.id_structures.get(drug_id, set())
            if canon and (not allowed or canon not in allowed):
                raise ValueError(f'Name/ID and supplied structure do not match: {name} ({drug_id})')
            # Identity determines the reproducible MC sequence; it is not tied
            # to whether the drug already has a frozen PCA row.
            if not canon and drug_id not in self.features.index:
                if len(allowed) != 1:
                    raise ValueError(f'Supply a validated structure for registered drug: {drug_id}')
                canon = next(iter(allowed))
            return (drug_id, canon)
        if not canon:
            raise ValueError(f'Unrecognized drug requires a SMILES string: {name}')
        matches = self.structures.get(canon, set())
        if len(matches) == 1:
            return (next(iter(matches)), canon)
        if len(matches) > 1:
            raise ValueError(f'Structure maps to multiple IDs; supply one explicitly: {sorted(matches)}')
        return ('SMILES:' + hashlib.sha256(canon.encode('utf-8')).hexdigest(), canon)

    def _references(self):
        if self.reference_fps is None:
            folder = self.root / 'data/DrugBank5.0_Approved_drugs'
            paths = {p.stem: p for p in folder.iterdir() if p.is_file()}
            missing = set(self.reference_ids) - set(paths)
            if missing:
                raise ValueError(f'Missing reference structures: {sorted(missing)[:10]}')
            with rdkit_log_block():
                self.reference_fps = [fingerprint(load_reference_molecule(paths[k])) for k in self.reference_ids]
        return self.reference_fps

    def validate_feature_path(self, tolerance=1e-6):
        if not np.isfinite(tolerance) or not 0 < tolerance <= 1e-6:
            raise ValueError('Feature tolerance must be positive and no larger than 1e-6')
        if self.feature_validation is not None:
            if self.feature_validation['max_abs_error'] > tolerance:
                raise ValueError('Cached feature reconstruction exceeds the requested tolerance')
            return {**self.feature_validation, 'tolerance': tolerance}
        from rdkit import rdBase
        from threadpoolctl import threadpool_limits
        references = self._references()
        target = self.features.loc[self.reference_ids].to_numpy(dtype=np.float64)
        errors = np.empty(len(references), dtype=np.float64)
        with threadpool_limits(limits=4):
            for start in range(0, len(references), 128):
                profile = np.asarray([DataStructs.BulkTanimotoSimilarity(fp, references)
                                      for fp in references[start:start + 128]], dtype=np.float64)
                projection = (profile - self.mean) @ self.components.T
                errors[start:start + len(projection)] = np.max(np.abs(projection - target[start:start + len(projection)]), axis=1)
        report = {'protocol': FEATURE_PROTOCOL, 'rdkit': rdBase.rdkitVersion,
                  'reference_count': len(references), 'tolerance': tolerance,
                  'max_abs_error': float(errors.max()),
                  'median_row_max_abs_error': float(np.median(errors)),
                  'failed_count': int((errors > tolerance).sum())}
        if not np.isfinite(errors).all() or report['failed_count']:
            bad = [self.reference_ids[i] for i in np.flatnonzero(errors > tolerance)[:10]]
            raise ValueError(f'Structure feature reconstruction failed; new structure prediction is blocked: {report}; IDs={bad}')
        report['status'] = 'PASS'
        self.feature_validation = report
        return dict(report)

    def project_molecule(self, mol):
        self.validate_feature_path()
        with rdkit_log_block():
            fp = fingerprint(mol)
            profile = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, self._references()), dtype=np.float64)
        projection = (profile - self.mean) @ self.components.T
        if not np.isfinite(projection).all():
            raise ValueError('Non-finite structure projection')
        return projection

    def featurize(self, identities: dict[str, str]) -> tuple[pd.DataFrame, dict[str, str]]:
        extra, sources = ({}, {})
        for drug_id, canon in identities.items():
            if drug_id in self.features.index:
                sources[drug_id] = 'frozen_pca50'
                continue
            if not canon:
                raise ValueError(f'No structure for: {drug_id}')
            extra[drug_id] = self.project_molecule(Chem.MolFromSmiles(canon))
            sources[drug_id] = FEATURE_PROTOCOL
        result = self.features.copy()
        if extra:
            result = pd.concat([result, pd.DataFrame.from_dict(extra, orient='index', columns=result.columns)])
        return (result, sources)


def complete_directions(samples: pd.DataFrame) -> pd.DataFrame:
    """Add missing reverse records within each prescription.

    Original rows and sample IDs are preserved. A generated reverse record
    exchanges the drug IDs and names, and receives a deterministic sample ID.
    Calling this function again does not add further records.
    """
    rows = samples.to_dict('records')
    keys = {
        (str(row['Prescription']), row['drug_a_id'], row['drug_b_id'])
        for row in rows
    }
    used_ids = {str(row['sample_id']) for row in rows}
    reverse_rows = []

    for row in rows:
        reverse_key = (
            str(row['Prescription']), row['drug_b_id'], row['drug_a_id']
        )
        if reverse_key in keys:
            continue

        sample_id = '__deepddix_reverse__' + json.dumps(
            reverse_key, separators=(',', ':'), ensure_ascii=True
        )
        if sample_id in used_ids:
            raise ValueError('Generated reverse sample ID collides with supplied ID')

        reverse_row = dict(row)
        reverse_row.update(
            sample_id=sample_id,
            drug_a_id=row['drug_b_id'],
            drug_b_id=row['drug_a_id'],
            drug_a_name=row['drug_b_name'],
            drug_b_name=row['drug_a_name'],
        )
        reverse_rows.append(reverse_row)
        keys.add(reverse_key)
        used_ids.add(sample_id)

    return pd.DataFrame(rows + reverse_rows, columns=samples.columns)


def prepare_input(input_file, root: str | Path=ROOT, pca_profile_file=None):
    frame = pd.read_csv(input_file, sep='\t', dtype=str, keep_default_na=False)
    frame.columns = [c.strip() for c in frame.columns]
    resolver = DrugResolver(root)
    identities, rows, mapping = ({}, [], [])
    if {'drug_a_id', 'drug_b_id'} <= set(frame.columns):
        for n, row in enumerate(frame.to_dict('records')):
            a, b = (row['drug_a_id'].strip(), row['drug_b_id'].strip())
            a, ca = resolver.resolve(a, row.get('drug_a_smiles', ''), a)
            b, cb = resolver.resolve(b, row.get('drug_b_smiles', ''), b)
            identities[a] = ca or identities.get(a, '')
            identities[b] = cb or identities.get(b, '')
            if a == b:
                raise ValueError(f'Self pair at input row {n + 2}')
            rows.append({'sample_id': row.get('sample_id') or f'pair_{n + 1}', 'Prescription': row.get('Prescription', '1'), 'drug_a_id': a, 'drug_b_id': b, 'drug_a_name': a, 'drug_b_name': b})
        if len({r['sample_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate sample_id in pair input')
    else:
        required = {'Prescription', 'Drug name', 'Smiles'}
        if not required <= set(frame.columns):
            raise ValueError('Input needs Prescription, Drug name, Smiles, or drug_a_id and drug_b_id columns')
        groups = {}
        for n, row in enumerate(frame.to_dict('records')):
            group, name, smiles = (row['Prescription'].strip(), row['Drug name'].strip(), row['Smiles'].strip())
            if not group or not name:
                raise ValueError(f'Empty prescription or name on row {n + 2}')
            drug, canon = resolver.resolve(name, smiles, row.get('DrugBank ID', ''))
            identities[drug] = canon or identities.get(drug, '')
            group_rows = groups.setdefault(group, [])
            if drug in {r[0] for r in group_rows}:
                raise ValueError(f'Duplicate drug identity in prescription {group}: {drug}')
            group_rows.append((drug, name))
            mapping.append({'Prescription': group, 'input_name': name, 'drug_id': drug, 'canonical_smiles': canon})
        for group, items in groups.items():
            for a, b in itertools.combinations(items, 2):
                for left, right in [(a, b), (b, a)]:
                    sid = json.dumps([group, left[0], right[0]], ensure_ascii=True, separators=(',', ':'))
                    rows.append({'sample_id': sid, 'Prescription': group, 'drug_a_id': left[0], 'drug_b_id': right[0], 'drug_a_name': left[1], 'drug_b_name': right[1]})
    if not rows:
        raise ValueError('No drug pairs were generated')
    features, sources = resolver.featurize(identities)
    if pca_profile_file:
        supplied = pd.read_csv(pca_profile_file, index_col=0)
        if supplied.shape[1] != 50 or not supplied.index.is_unique:
            raise ValueError('PCA input needs 50 columns and unique stable drug IDs')
        for drug in identities:
            if drug not in supplied.index:
                raise ValueError(f'PCA input missing stable ID: {drug}')
            vals = supplied.loc[drug].to_numpy(dtype=np.float32)
            if not np.isfinite(vals).all():
                raise ValueError(f'Invalid PCA row: {drug}')
            expected = features.loc[drug].to_numpy(dtype=np.float32)
            if not np.array_equal(vals, expected):
                raise ValueError(f'Supplied PCA row differs from the validated model feature: {drug}')
            features.loc[drug] = vals
            sources[drug] = 'supplied_pca50'
    if not mapping:
        mapping = [{'Prescription': '', 'input_name': k, 'drug_id': k, 'canonical_smiles': v} for k, v in identities.items()]
    for row in mapping:
        row['feature_source'] = sources[row['drug_id']]
    return (pd.DataFrame(rows), features, pd.DataFrame(mapping))
