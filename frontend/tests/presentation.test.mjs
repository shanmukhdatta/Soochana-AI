import fs from 'node:fs';import assert from 'node:assert/strict';import {test} from 'node:test';
const app=fs.readFileSync(new URL('../src/App.tsx',import.meta.url),'utf8');
const css=fs.readFileSync(new URL('../src/mono.css',import.meta.url),'utf8');
test('homepage has a real introduction and working dashboard navigation in source',()=>{for(const s of ['function Home(',"setPage('events')",'onTab={setPage}','Core technical requirements','Additional implementation'])assert.ok(app.includes(s),s)});
test('two view modes share selection and results state rather than mounting duplicate apps',()=>{assert.ok(app.includes("useState<ViewMode>('quick')"));assert.ok(app.includes('setSelected(s);setPage('));assert.ok(app.includes('setRebalance={setRebalance} mode={mode}'));assert.ok(app.includes('onMode={setMode}'));assert.ok(app.includes('clearProposal={()=>setProposal(null)} mode={mode}'));});
test('advanced actions remain in Analyst View',()=>{for(const s of ['Run holdout backtest','Run what-if','Run reverse stress','MANUAL AI INFERENCE','probabilities','onRebalance','scenarioForExecution'])assert.ok(app.includes(s),s)});
test('quick mode simplifies but preserves analyst stress controls',()=>{assert.ok(app.includes("mode==='analyst'&&<section className=\"panel mechanisms\""));assert.ok(app.includes("mode==='analyst'&&<section className=\"panel additional-stress\""));assert.ok(css.includes('.mode-quick .shock-fields{display:none}'));assert.ok(css.includes('.mode-quick .probabilities{display:none}'));});
test('bujji financial assistant is mounted with welcome message, starters and disclaimer',()=>{
  assert.ok(app.includes('function BujjiAssistant('));
  assert.ok(app.includes("Hi! I'm Bujji, your RiskPulse AI guide"));
  assert.ok(app.includes('What is Expected Credit Loss?'));
  assert.ok(app.includes('AI-generated explanations may contain errors'));
  assert.ok(css.includes('.bujji-launcher'));
  assert.ok(css.includes('.bujji-panel'));
});
