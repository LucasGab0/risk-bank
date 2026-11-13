"""Verificação da interface em Chromium. Executar separadamente do pytest."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]


async def run_browser(url, executable, screenshot_dir):
    results, errors = [], []
    async with async_playwright() as playwright:
        options = {'headless': True}
        if executable:
            options['executable_path'] = executable
        browser = await playwright.chromium.launch(**options)
        page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        await page.goto(url)
        await page.wait_for_function("!document.getElementById('run').disabled")
        assert await page.locator('#model').input_value() == 'xgboost'
        assert await page.locator('#model option').count() == 3
        results.append('Inicialização e catálogo dos três modelos')
        model_results = {}
        for key in ['logistic', 'mlp', 'xgboost']:
            await page.locator('#model').select_option(key)
            values = []
            for preset in ['good', 'bad']:
                async with page.expect_response(lambda response: response.url.endswith('/predict') and response.status == 200) as response_info:
                    await page.locator(f'[data-preset={preset}]').click()
                response = await response_info.value
                data = await response.json()
                await page.wait_for_function("document.getElementById('pct').textContent.includes('%')")
                assert data['model_id'] == key
                assert await page.locator('#fx li').count() == 3
                assert 'Calibração:' in await page.locator('#model-metrics').inner_text()
                values.append(data['prob'])
            assert values[0] < values[1]
            model_results[key] = {'low': values[0], 'high': values[1]}
        results.append('Previsão real e ordenação dos exemplos nos três modelos')
        assert await page.locator('#hist li').count() == 5
        results.append('Histórico limitado a cinco análises')
        probability = await page.locator('#pct').inner_text()
        await page.locator('#thr').fill('100')
        assert await page.locator('#verdict').inner_text() == 'Provável adimplente'
        await page.locator('#thr').fill('0')
        assert await page.locator('#verdict').inner_text() == 'Provável inadimplente'
        assert await page.locator('#pct').inner_text() == probability
        results.append('Limiar altera classificação sem alterar probabilidade')
        await page.locator('#age').fill('40')
        assert await page.locator('#pct').inner_text() == '–'
        assert await page.locator('#fx li').count() == 0
        await page.locator('#limit').fill('0')
        await page.locator('#run').click()
        assert await page.locator('#pct').inner_text() == '–'
        assert not await page.locator('#limit').evaluate('(element) => element.checkValidity()')
        await page.locator('#limit').fill('30000')
        await page.locator('#bill0').fill('-3000')
        assert await page.locator('#chart rect').evaluate_all('(elements) => elements.every(e => Number(e.getAttribute("height")) >= 0)')
        results.append('Edição invalida resultado, validação de formulário e faturas negativas')
        await page.locator('#abrir-ajuda').click()
        assert await page.locator('#ajuda').is_visible()
        await page.get_by_role('button', name='Fechar', exact=True).click()
        results.append('Diálogo de ajuda')
        await page.route('**/predict', lambda route: route.abort('failed'))
        await page.locator('#run').click()
        await page.wait_for_function("document.getElementById('status').classList.contains('error')")
        assert await page.locator('#pct').inner_text() == '–'
        assert await page.locator('#verdict').inner_text() == 'Análise indisponível'
        await page.unroute('**/predict')
        results.append('Falha de rede sem previsão local')
        await page.route('**/predict', lambda route: route.fulfill(status=503, content_type='application/json', body='{"detail":"indisponível"}'))
        await page.locator('#run').click()
        await page.wait_for_function("document.getElementById('status').textContent.includes('Modelo indisponível')")
        assert await page.locator('#pct').inner_text() == '–'
        await page.unroute('**/predict')
        results.append('Modelo indisponível sem resultado fictício')
        pending = asyncio.Event()
        release = asyncio.Event()
        async def delayed(route):
            pending.set()
            await release.wait()
            try:
                await route.continue_()
            except Exception:
                pass  # O navegador cancela a requisição após a edição.
        await page.route('**/predict', delayed)
        await page.locator('#run').click()
        await asyncio.wait_for(pending.wait(), timeout=10)
        await page.locator('#age').fill('41')
        release.set()
        await page.wait_for_function("!document.getElementById('run').disabled")
        assert await page.locator('#pct').inner_text() == '–'
        await page.unroute('**/predict')
        results.append('Resposta pendente cancelada quando os dados mudam')
        await page.locator('[data-preset=bad]').click()
        await page.wait_for_function("document.getElementById('pct').textContent.includes('%')")
        if screenshot_dir:
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            await page.evaluate("document.documentElement.style.scrollBehavior = 'auto'; window.scrollTo(0, 0)")
            await page.screenshot(path=str(screenshot_dir / 'painel_desktop.png'), full_page=True, animations='disabled')
        await page.set_viewport_size({'width': 390, 'height': 844})
        await page.locator('#model').select_option('logistic')
        await page.locator('[data-preset=good]').click()
        await page.wait_for_function("document.getElementById('pct').textContent.includes('%')")
        assert await page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        assert await page.locator('.scroll').evaluate('(element) => element.scrollWidth > element.clientWidth')
        if screenshot_dir:
            await page.evaluate('window.scrollTo(0, 0)')
            await page.screenshot(path=str(screenshot_dir / 'painel_celular.png'), full_page=True, animations='disabled')
        results.append('Layout de celular e rolagem do histórico mensal')
        await page.route('**/models', lambda route: route.fulfill(status=503, content_type='application/json', body='{"detail":"indisponível"}'))
        await page.reload()
        await page.wait_for_function("document.getElementById('status').classList.contains('error')")
        assert await page.locator('#run').is_disabled()
        assert await page.locator('#pct').inner_text() == '–'
        results.append('Inicialização sem modelos mantém análise indisponível')
        assert not errors, errors
        report = {'passed': len(results), 'scenarios': results, 'model_examples': model_results,
                  'javascript_errors': errors, 'browser': browser.version}
        await browser.close()
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chromium', help='Caminho opcional para Chromium instalado no sistema')
    parser.add_argument('--screenshots', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'server:app', '--host', '127.0.0.1', '--port', str(port)],
                                   cwd=ROOT, stdout=log, stderr=log, env={**os.environ, 'OPENBLAS_NUM_THREADS':'2', 'OMP_NUM_THREADS':'2'})
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    with urlopen(url + '/health', timeout=1) as response:
                        if json.load(response)['model_loaded']:
                            break
                except OSError:
                    pass
                if process.poll() is not None or time.monotonic() > deadline:
                    log.seek(0)
                    raise RuntimeError(log.read().decode())
                time.sleep(0.1)
            report = asyncio.run(run_browser(url, args.chromium, args.screenshots))
            rendered = json.dumps(report, ensure_ascii=False, indent=2)
            print(rendered)
            if args.report:
                args.report.write_text(rendered, encoding='utf-8')
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
