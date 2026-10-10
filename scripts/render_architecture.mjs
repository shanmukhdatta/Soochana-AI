import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const chromePath = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const htmlPath = path.resolve('scripts/architecture_diagram.html');
const outPath = path.resolve('docs/architecture.png');

console.log(`Rendering ${htmlPath} to ${outPath}...`);
const cmd = `"${chromePath}" --headless=new --disable-gpu --screenshot="${outPath}" --window-size=1600,1050 "file:///${htmlPath.replace(/\\/g, '/')}"`;
execSync(cmd, { stdio: 'inherit' });

if (fs.existsSync(outPath)) {
  const stat = fs.statSync(outPath);
  console.log(`Successfully generated docs/architecture.png (${(stat.size / 1024).toFixed(1)} KB)`);
} else {
  console.error('Failed to generate architecture.png');
  process.exit(1);
}
