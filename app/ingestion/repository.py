"""SQLite storage: bounded queries, atomic claims, retryable failures, source provenance."""
from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime,timedelta,timezone
from pathlib import Path
from .records import RawItem,iso,utc_now,sha

SCHEMA = '''
CREATE TABLE IF NOT EXISTS signals (
 id TEXT PRIMARY KEY, source_type TEXT NOT NULL, source_name TEXT NOT NULL,
 source_id TEXT NOT NULL, source_url TEXT NOT NULL, ticker TEXT NOT NULL,
 match_method TEXT NOT NULL, text TEXT NOT NULL, first_seen_at TEXT NOT NULL,
 reported_at TEXT, timestamp_quality TEXT NOT NULL, analysis_as_of TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
 next_retry_at TEXT, claimed_at TEXT, last_error TEXT, inference_json TEXT,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 UNIQUE(source_type,source_id,ticker)
);
CREATE INDEX IF NOT EXISTS idx_signals_processing ON signals(status,next_retry_at,created_at);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker,created_at);
CREATE TABLE IF NOT EXISTS collection_runs (
 id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL, started_at TEXT NOT NULL,
 status TEXT NOT NULL, fetched INTEGER NOT NULL, inserted INTEGER NOT NULL,
 rejected INTEGER NOT NULL, detail TEXT
);
'''


class Repository:
    def __init__(self, path:Path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connection() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connection(self):
        db=sqlite3.connect(str(self.path),timeout=15,isolation_level=None)
        try:
            db.row_factory=sqlite3.Row
            db.execute('PRAGMA busy_timeout=15000')
            db.execute('PRAGMA journal_mode=WAL')
            yield db
        finally:
            db.close()

    def insert(self,item:RawItem,ticker:str,match_method:str) -> bool:
        key=sha('|'.join((item.source_type,item.source_id,ticker)))
        now=iso(utc_now())
        # Never use provider-created or source-reported dates to backdate inference.
        observed=iso(item.first_seen_at)
        with self.connection() as db:
            cursor=db.execute('''INSERT OR IGNORE INTO signals
            (id,source_type,source_name,source_id,source_url,ticker,match_method,text,
             first_seen_at,reported_at,timestamp_quality,analysis_as_of,status,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (key,item.source_type,item.source_name,item.source_id,item.source_url,ticker,
             match_method,item.text,observed,iso(item.reported_at) if item.reported_at else None,
             item.timestamp_quality,observed,'pending',now,now))
            return cursor.rowcount==1

    def record_collection(self,source:str,status:str,fetched:int,inserted:int,rejected:int,detail:str|None=None):
        with self.connection() as db:
            db.execute('INSERT INTO collection_runs(source,started_at,status,fetched,inserted,rejected,detail) VALUES(?,?,?,?,?,?,?)',
                       (source,iso(utc_now()),status,fetched,inserted,rejected,(detail or '')[:500]))

    def claim(self,limit:int,lease_seconds:int,max_attempts:int) -> list[dict]:
        """BEGIN IMMEDIATE makes concurrent worker claims mutually exclusive."""
        now=utc_now()
        cutoff=iso(now-timedelta(seconds=lease_seconds))
        now_str=iso(now)
        with self.connection() as db:
            try:
                db.execute('BEGIN IMMEDIATE')
                # Orphaned work due to process death can be reclaimed.
                db.execute('''UPDATE signals SET status='retry',claimed_at=NULL,updated_at=?
                  WHERE status='processing' AND claimed_at < ? AND attempts < ?''',
                    (now_str,cutoff,max_attempts))
                db.execute('''UPDATE signals SET status='dead',last_error='Processing lease expired after maximum attempts',
                  updated_at=? WHERE status='processing' AND claimed_at < ? AND attempts >= ?''',
                  (now_str,cutoff,max_attempts))
                rows=db.execute('''SELECT * FROM signals WHERE
                     status='pending' OR (status='retry' AND (next_retry_at IS NULL OR next_retry_at<=?))
                     ORDER BY first_seen_at ASC,id ASC LIMIT ?''',(now_str,limit)).fetchall()
                ids=[row['id'] for row in rows]
                for key in ids:
                    db.execute('''UPDATE signals SET status='processing',attempts=attempts+1,
                        claimed_at=?,updated_at=? WHERE id=?''',(now_str,now_str,key))
                db.execute('COMMIT')
            except Exception:
                db.execute('ROLLBACK')
                raise
        return [dict(x) for x in rows]

    def success(self,key:str,prediction:dict):
        with self.connection() as db:
            db.execute('''UPDATE signals SET status='completed',inference_json=?,last_error=NULL,
                claimed_at=NULL,updated_at=? WHERE id=? AND status='processing' ''',
                (json.dumps(prediction,allow_nan=False),iso(utc_now()),key))

    def failure(self,key:str,error:str,attempts:int,max_attempts:int,retry_base_seconds:int,permanent=False):
        dead=permanent or attempts>=max_attempts
        next_try=None if dead else iso(utc_now()+timedelta(seconds=retry_base_seconds*(2**max(0,attempts-1))))
        with self.connection() as db:
            db.execute('''UPDATE signals SET status=?,last_error=?,claimed_at=NULL,
                         next_retry_at=?,updated_at=? WHERE id=? AND status='processing' ''',
                       ('dead' if dead else 'retry',error[:500],next_try,iso(utc_now()),key))

    def list(self,limit=30,offset=0,ticker=None,status=None,source_type=None):
        terms=[];params=[]
        for name,value in [('ticker',ticker),('status',status),('source_type',source_type)]:
            if value is not None: terms.append(f'{name}=?');params.append(value)
        where=' WHERE '+' AND '.join(terms) if terms else ''
        with self.connection() as db:
            total=db.execute('SELECT COUNT(*) FROM signals'+where,params).fetchone()[0]
            rows=db.execute('SELECT * FROM signals'+where+' ORDER BY first_seen_at DESC,id DESC LIMIT ? OFFSET ?',
                            [*params,limit,offset]).fetchall()
        return {'total':total,'limit':limit,'offset':offset,'items':[self._present(dict(x)) for x in rows]}

    def one(self,key:str):
        with self.connection() as db:
            row=db.execute('SELECT * FROM signals WHERE id=?',(key,)).fetchone()
        return self._present(dict(row)) if row else None

    @staticmethod
    def _present(x:dict):
        x['inference']=json.loads(x.pop('inference_json')) if x.get('inference_json') else None
        x['published_at'] = x['reported_at'] if x['timestamp_quality']=='self_reported' else None
        x['provider_discovered_at'] = x['reported_at'] if x['timestamp_quality']=='gdelt_discovery' else None
        return x

    def summary(self):
        with self.connection() as db:
            status=[dict(r) for r in db.execute('SELECT status,COUNT(*) as count FROM signals GROUP BY status')]
            sources=[dict(r) for r in db.execute('SELECT source_type,COUNT(*) as count FROM signals GROUP BY source_type')]
            tickers=[dict(r) for r in db.execute('SELECT ticker,COUNT(*) as count FROM signals WHERE status="completed" GROUP BY ticker ORDER BY count DESC')]
            recent=[dict(r) for r in db.execute('SELECT source,started_at,status,fetched,inserted,rejected,detail FROM collection_runs ORDER BY id DESC LIMIT 20')]
        return {'status_counts':status,'source_counts':sources,'analyzed_by_ticker':tickers,'recent_collection_runs':recent}
