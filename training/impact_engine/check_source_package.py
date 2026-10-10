"""Verify that all three notebooks and Python exports are present and syntactically parse."""
import ast, json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
for stem,record in manifest.items():
    notebook=root/'notebooks'/f'{stem}.ipynb'
    script=root/'scripts'/f'{stem}.py'
    raw=json.loads(notebook.read_text(encoding='utf-8'))
    assert len(raw['cells']) == record['cell_count'],stem
    assert all(not c.get('outputs') and c.get('execution_count') is None for c in raw['cells'] if c['cell_type']=='code'),stem
    ast.parse(script.read_text(encoding='utf-8'),filename=str(script))
    print('SOURCE OK:',stem)
print('All notebook structures and Python source syntax checks passed. No training executed.')
