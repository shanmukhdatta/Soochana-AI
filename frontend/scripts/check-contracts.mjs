import fs from 'node:fs';import path from 'node:path';import assert from 'node:assert/strict';
const root=process.cwd();const api=fs.readFileSync(path.join(root,'src/api.ts'),'utf8');const app=fs.readFileSync(path.join(root,'src/App.tsx'),'utf8');const css=fs.readFileSync(path.join(root,'src/styles.css'),'utf8');
const routes=['/health','/ingestion/health','/signals/summary','/signals','/analyze','/module-a/index','/ingestion/module-a/recommend','/module-a/backtest','/module-b/portfolio','/module-b/scenarios','/module-b/stress','/module-b/what-if','/module-b/reverse-stress','/module-b-suggestion'];
for(const route of routes){assert.ok(api.includes(route),`Missing documented backend route: ${route}`)}
for(const name of ['Risk Journey Explorer','EVENT STREAM','ALLOCATION SHIFT ANALYZER','SCENARIO DESIGN','FINBERT','XGBOOST']){assert.ok(app.toLowerCase().includes(name.toLowerCase()),`Missing Figma design content: ${name}`)}
for(const token of ['--bg:#10121c','--iris:#9b8cff','--amber:#f0b56b'])assert.ok(css.includes(token),`Missing Figma palette token ${token}`);
assert.ok(!api.includes('MARKETAUX_API_TOKEN'));assert.ok(!api.includes('INGESTION_ADMIN_TOKEN'));
console.log(`Contract and visual source checks passed: ${routes.length} paths, 6 screen markers, 3 theme colors.`);
