"""Safe, idempotent router registration. Preserves Main/Module A/Risk Engine."""
from pathlib import Path
import ast
import shutil

root=Path(__file__).resolve().parent
main=root/'app'/'main.py'
if not main.is_file():
    raise SystemExit('ERROR: app/main.py missing. Copy add-on files into the existing c2c project root first.')
if not (root/'app'/'module_b'/'router.py').is_file():
    raise SystemExit('ERROR: app/module_b/router.py missing. Copy folder contents into app/module_b first.')
content=main.read_text(encoding='utf-8')
try:ast.parse(content)
except SyntaxError as exc:raise SystemExit(f'ERROR: existing app/main.py has syntax error: {exc}') from exc
marker='app.include_router(module_b_router)'
if marker in content:
    print('Module B already registered; no changes made.')
else:
    if 'app = FastAPI(' not in content:
        raise SystemExit('ERROR: FastAPI app assignment not found; review app/main.py manually.')
    if 'module_b_router' in content:
        raise SystemExit('ERROR: Partial Module B registration found; review app/main.py manually.')
    insertion=("\n\n# RiskPulse Module B; original Risk Engine and Module A remain unchanged.\n"
               "from app.module_b.router import router as module_b_router\n"
               "app.include_router(module_b_router)\n")
    candidate=content+insertion
    ast.parse(candidate)
    backup=root/'app'/'main.py.before_module_b.bak'
    if not backup.exists():shutil.copy2(main,backup)
    main.write_text(candidate,encoding='utf-8')
    print('Registered Module B router, backup:',backup)
print('Next: python -m pytest -q tests -k module_b')
