import { execSync } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';

// Set CHROME_PATH to an installed Chrome/Chromium executable if not on PATH.
// Screenshots are stored in a repository-local ignored folder by default.
const ARTIFACT_DIR = path.resolve(process.env.AUDIT_SCREENSHOT_DIR || path.join('frontend', 'audit_screenshots'));
const CHROME_PATH = process.env.CHROME_PATH || 'chrome';

if (!fs.existsSync(ARTIFACT_DIR)) {
  fs.mkdirSync(ARTIFACT_DIR, { recursive: true });
}

const targets = [
  // 8 Required Product Views
  { name: 'audit_01_introduction_home.png', url: 'http://127.0.0.1:5173/#home', size: '1440,900' },
  { name: 'audit_02_signals_quick_view.png', url: 'http://127.0.0.1:5173/?mode=quick#events', size: '1440,900' },
  { name: 'audit_03_signals_analyst_view.png', url: 'http://127.0.0.1:5173/?mode=analyst#events', size: '1440,900' },
  { name: 'audit_04_module_a_quick_view.png', url: 'http://127.0.0.1:5173/?mode=quick#moduleA', size: '1440,900' },
  { name: 'audit_05_module_a_analyst_view.png', url: 'http://127.0.0.1:5173/?mode=analyst#moduleA', size: '1440,900' },
  { name: 'audit_06_module_b_quick_view.png', url: 'http://127.0.0.1:5173/?mode=quick#moduleB', size: '1440,900' },
  { name: 'audit_07_module_b_analyst_view.png', url: 'http://127.0.0.1:5173/?mode=analyst#moduleB', size: '1440,900' },
  { name: 'audit_08_bujji_assistant.png', url: 'http://127.0.0.1:5173/?mode=quick&bujji=open#events', size: '1440,900' },
  
  // 5 Responsive Breakpoint Views
  { name: 'audit_responsive_1600.png', url: 'http://127.0.0.1:5173/#home', size: '1600,1000' },
  { name: 'audit_responsive_1440.png', url: 'http://127.0.0.1:5173/?mode=quick#events', size: '1440,900' },
  { name: 'audit_responsive_1024.png', url: 'http://127.0.0.1:5173/?mode=analyst#events', size: '1024,768' },
  { name: 'audit_responsive_768.png', url: 'http://127.0.0.1:5173/?mode=quick#events', size: '768,1024' },
  { name: 'audit_responsive_390.png', url: 'http://127.0.0.1:5173/#home', size: '390,844' },
];

console.log('Capturing audit screenshots via headless Chrome...\n');
for (const t of targets) {
  const dest = path.join(ARTIFACT_DIR, t.name);
  console.log(`Capturing: ${t.name} (${t.size}) from ${t.url}`);
  try {
    const cmd = `"${CHROME_PATH}" --headless=new --disable-gpu --screenshot="${dest}" --window-size=${t.size} "${t.url}"`;
    execSync(cmd, { stdio: 'ignore', timeout: 15000 });
    const stat = fs.statSync(dest);
    console.log(`  ✔ Captured: ${t.name} (${(stat.size / 1024).toFixed(1)} KB)`);
  } catch (err) {
    console.error(`  ❌ Failed: ${t.name}:`, err.message);
  }
}
console.log('\nScreenshot capture attempt finished; inspect failures above.');
