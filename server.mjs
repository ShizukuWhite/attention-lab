import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const host = '127.0.0.1';
const port = 3000;
const productRoot = dirname(fileURLToPath(import.meta.url));
const publicFiles = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/index.html', ['index.html', 'text/html; charset=utf-8']],
  ['/src/app.js', ['src/app.js', 'text/javascript; charset=utf-8']],
  ['/src/tokenizer.js', ['src/tokenizer.js', 'text/javascript; charset=utf-8']],
  ['/src/styles.css', ['src/styles.css', 'text/css; charset=utf-8']],
]);

const server = createServer(async (request, response) => {
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    response.writeHead(405, {
      Allow: 'GET, HEAD',
      'Content-Type': 'text/plain; charset=utf-8',
      'X-Content-Type-Options': 'nosniff',
    });
    response.end('Method not allowed');
    return;
  }

  let pathname;
  try {
    pathname = new URL(request.url, `http://${host}:${port}`).pathname;
  } catch {
    response.writeHead(400, { 'Content-Type': 'text/plain; charset=utf-8' });
    response.end('Bad request');
    return;
  }

  const publicFile = publicFiles.get(pathname);
  if (!publicFile) {
    response.writeHead(404, {
      'Cache-Control': 'no-store',
      'Content-Type': 'text/plain; charset=utf-8',
      'X-Content-Type-Options': 'nosniff',
    });
    response.end('Not found');
    return;
  }

  try {
    const [relativePath, contentType] = publicFile;
    const body = await readFile(join(productRoot, relativePath));
    response.writeHead(200, {
      'Cache-Control': 'no-store',
      'Content-Length': body.byteLength,
      'Content-Type': contentType,
      'X-Content-Type-Options': 'nosniff',
      'Referrer-Policy': 'no-referrer',
      'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'none'",
    });
    response.end(request.method === 'HEAD' ? undefined : body);
  } catch {
    response.writeHead(500, {
      'Content-Type': 'text/plain; charset=utf-8',
      'X-Content-Type-Options': 'nosniff',
    });
    response.end('Unable to read the requested file');
  }
});

server.listen(port, host, () => {
  console.log(`Attention Lab is running at http://${host}:${port}`);
  console.log('Press Ctrl+C to stop the local server.');
});
