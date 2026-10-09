"""Idempotent installer: inject only the ingestion router into existing app/main.py."""
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
main=ROOT/'app'/'main.py'
if not main.exists():
    raise SystemExit('Run this installer from the existing c2c root after merging add-on files')
if not (ROOT/'app'/'ingestion'/'router.py').is_file():
    raise SystemExit('app/ingestion/router.py missing: merge the staged files first')
content=main.read_text(encoding='utf-8')
marker='# RiskPulse two-source ingestion router'
if marker in content:
    if content.count(marker)!=1:raise SystemExit('Duplicate ingestion installer markers; inspect app/main.py')
    print('Already registered: ingestion router (no changes)')
else:
    backup=main.with_name('main.py.before_ingestion.bak')
    if not backup.exists():shutil.copy2(main,backup)
    addition='\n\n'+marker+'\nfrom app.ingestion.router import router as ingestion_router\napp.include_router(ingestion_router)\n'
    main.write_text(content.rstrip()+addition,encoding='utf-8')
    print('Registered ingestion router in app/main.py; backup:',backup)
