import { spawn } from 'child_process';
import http from 'http';
import fs from 'fs';

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function getDebuggerUrl(port = 9222, retries = 25) {
  for (let i = 0; i < retries; i++) {
    try {
      const res = await new Promise((resolve, reject) => {
        const req = http.get(`http://127.0.0.1:${port}/json`, (resp) => {
          let data = '';
          resp.on('data', (c) => (data += c));
          resp.on('end', () => {
            try {
              resolve(JSON.parse(data));
            } catch (err) {
              reject(err);
            }
          });
        });
        req.on('error', reject);
      });
      if (res && res.length > 0) {
        // find page target
        const pageTarget = res.find((t) => t.type === 'page' && t.webSocketDebuggerUrl) || res[0];
        if (pageTarget && pageTarget.webSocketDebuggerUrl) {
          return pageTarget.webSocketDebuggerUrl;
        }
      }
    } catch (e) {
      // wait
    }
    await sleep(300);
  }
  throw new Error('Could not connect to Chrome DevTools port');
}

async function runHeadlessTest() {
  console.log('[*] Launching headless Google Chrome directly at http://localhost:8000 ...');
  const chrome = spawn(
    'google-chrome',
    [
      '--headless=new',
      '--no-sandbox',
      '--disable-gpu',
      '--remote-debugging-port=9222',
      '--remote-allow-origins=*',
      'http://localhost:8000',
    ],
    { stdio: 'ignore' }
  );

  try {
    const wsUrl = await getDebuggerUrl(9222);
    console.log('[*] Connected to Chrome DevTools WebSocket:', wsUrl);

    const ws = new WebSocket(wsUrl);
    await new Promise((resolve, reject) => {
      ws.onopen = resolve;
      ws.onerror = reject;
    });

    let id = 1;
    const send = (method, params = {}) => {
      const msgId = id++;
      return new Promise((resolve) => {
        const listener = (event) => {
          const data = JSON.parse(event.data);
          if (data.id === msgId) {
            ws.removeEventListener('message', listener);
            resolve(data.result);
          }
        };
        ws.addEventListener('message', listener);
        ws.send(JSON.stringify({ id: msgId, method, params }));
      });
    };

    const consoleLogs = [];
    const errors = [];
    ws.addEventListener('message', (event) => {
      const data = JSON.parse(event.data);
      if (data.method === 'Runtime.consoleAPICalled') {
        const text = data.params.args.map((a) => a.value || a.description || '').join(' ');
        consoleLogs.push(`[Console ${data.params.type}]: ${text}`);
      }
      if (data.method === 'Runtime.exceptionThrown') {
        errors.push(data.params.exceptionDetails);
      }
    });

    await send('Runtime.enable');
    await send('Page.enable');

    console.log('[*] Waiting for React application to mount and render data...');
    await sleep(5000);

    // Evaluate rendered page state
    const evalRes = await send('Runtime.evaluate', {
      expression: `(() => {
        const title = document.title;
        const root = document.getElementById('root');
        const header = document.querySelector('header');
        const buttons = Array.from(document.querySelectorAll('button')).map(b => b.innerText.trim()).filter(Boolean);
        const stagePills = Array.from(document.querySelectorAll('button')).map(b => b.innerText.trim()).filter(t => t.includes('.'));
        const scenariosSelect = document.querySelector('select');
        const selectedScenario = scenariosSelect ? scenariosSelect.value : null;
        const bodySnippet = document.body.innerText.replace(/\\s+/g, ' ').slice(0, 400);
        return {
          title,
          hasRootChildren: root ? root.children.length > 0 : false,
          hasHeader: !!header,
          buttonsCount: buttons.length,
          stagePills,
          selectedScenario,
          bodySnippet,
        };
      })()`,
      returnByValue: true,
    });

    console.log('[*] Page state evaluation:');
    console.log(JSON.stringify(evalRes?.result?.value, null, 2));

    if (consoleLogs.length > 0) {
      console.log('[*] Captured browser console messages:');
      consoleLogs.forEach((l) => console.log('   ', l));
    }

    // Capture screenshot
    const shot = await send('Page.captureScreenshot', { format: 'png' });
    if (shot && shot.data) {
      fs.writeFileSync('headless_screenshot.png', Buffer.from(shot.data, 'base64'));
      console.log('[+] Screenshot saved to headless_screenshot.png (size: ' + shot.data.length + ' bytes)');
    }

    if (errors.length > 0) {
      console.error('[!] Runtime exceptions found:', errors);
      process.exit(1);
    } else {
      console.log('[+] Headless instance verification PASSED cleanly!');
    }

    ws.close();
  } finally {
    chrome.kill('SIGKILL');
  }
}

runHeadlessTest().catch((err) => {
  console.error('[!] Headless test script error:', err);
  process.exit(1);
});
