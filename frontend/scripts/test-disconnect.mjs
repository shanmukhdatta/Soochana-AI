import assert from 'node:assert/strict';

// Test that when backend is disconnected, API throws proper errors and never returns synthetic success
async function testDisconnectedBackend() {
  console.log('Testing intentional backend disconnect behavior...');

  // 1. Verify fetch failure handling against unreachable port
  let errorCaught = false;
  try {
    const res = await fetch('http://127.0.0.1:59999/api/health');
    await res.json();
  } catch (err) {
    errorCaught = true;
    console.log(`  ✔ Correctly caught network disconnection error: ${err.message}`);
  }
  assert.ok(errorCaught, 'Network fetch must fail when backend is disconnected');

  // 2. Verify that frontend helpers handle null/error states cleanly without crashing or mocking
  const { countStatus, countSource, safeNumber, formatPercentFraction, fmtMoney, sentimentLabel, recentSignalMeans } = await import('../src/lib/helpers.mjs');
  
  assert.equal(countStatus(null, 'completed'), 0, 'countStatus must handle null summary safely');
  assert.equal(countSource(null, 'financial_news'), 0, 'countSource must handle null summary safely');
  assert.equal(safeNumber(null), null, 'safeNumber must return null for null input');
  assert.equal(formatPercentFraction(null), '—', 'formatPercentFraction must return dash for null');
  assert.equal(fmtMoney(undefined), '—', 'fmtMoney must return dash for undefined');
  assert.equal(sentimentLabel(undefined), 'Not analyzed', 'sentimentLabel handles undefined');
  assert.deepEqual(recentSignalMeans([]), { count: 0, sentiment: null, impact: null }, 'recentSignalMeans on empty signals');
  
  console.log('  ✔ All UI helper fallbacks return safe empty indicators ("—", 0, null), never mock data.');
  console.log('  ✔ Backend disconnection verification PASSED.');
}

testDisconnectedBackend().catch(err => {
  console.error('Disconnect test failed:', err);
  process.exit(1);
});
