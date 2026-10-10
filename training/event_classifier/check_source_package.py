"""Static integrity check of Event Classifier training package; NOT a model training test."""
from pathlib import Path
import ast, json, hashlib
ROOT = Path(__file__).resolve().parent
LABELS = {str(i):name for i,name in enumerate(['Geopolitical','Macroeconomic','Credit Event','Merger / Acquisition','Product Launch'])}
assert json.loads((ROOT/'event_labels.json').read_text(encoding='utf-8')) == LABELS
manifest = json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))
for name,item in manifest['sources'].items():
    nb=json.loads((ROOT/item['notebook']).read_text(encoding='utf-8'))
    assert len(nb['cells'])==item['cells']
    assert sum(c['cell_type']=='code' for c in nb['cells'])==item['code_cells']
    assert all(c.get('outputs',[]) == [] and c.get('execution_count') is None for c in nb['cells'] if c['cell_type']=='code')
    body=''.join(''.join(c['source']) for c in nb['cells'])
    assert hashlib.sha256(body.encode('utf-8')).hexdigest() == item['source_cells_sha256']
    ast.parse((ROOT/item['script']).read_text(encoding='utf-8'), filename=item['script'])
print('PASS: source integrity, class taxonomy and Python export syntax (4 stages).')
print('This check does NOT train models, download datasets or confirm runtime checkpoint identity.')
