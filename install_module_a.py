"""SAFE and idempotent Module A router registration in existing c2c app."""
from pathlib import Path

root = Path(__file__).resolve().parent
main = root / 'app' / 'main.py'
if not main.is_file():
    raise SystemExit('ERROR: app/main.py missing. Extract inside the existing c2c root.')
content = main.read_text(encoding='utf-8')
marker = 'app.include_router(module_a_router)'
if marker in content:
    print('Module A already registered; no changes made.')
else:
    if 'app = FastAPI(' not in content:
        raise SystemExit('ERROR: FastAPI app assignment not found; review app/main.py.')
    backup = root / 'app' / 'main.py.before_module_a.bak'
    if not backup.exists():
        backup.write_text(content, encoding='utf-8')
    insertion = (
        '\n\n# RiskPulse Module A; original Risk Engine endpoints remain unchanged.\n'
        'from app.module_a.router import router as module_a_router\n'
        'app.include_router(module_a_router)\n'
    )
    main.write_text(content + insertion, encoding='utf-8')
    print(f'Registered Module A. Backup: {backup}')
print('Next: python -m pytest -q tests -k module_a')
