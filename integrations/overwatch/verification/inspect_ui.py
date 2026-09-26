"""Inspect the REAL running Overwatch React UI; never serve a substitute dashboard.

Start Overwatch with isolated cache/registry and OVERWATCH_LOCAL_MODELS_ONLY=1.
Requires Playwright plus an installed browser. All requests target loopback.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    u = urlsplit(args.url)
    if u.scheme != 'http' or u.hostname not in {'127.0.0.1', 'localhost', '::1'} or u.username or u.password:
        raise ValueError('only explicit loopback Overwatch inspection is supported')
    if args.out.exists(): raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1440, 'height': 1100})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until='networkidle')
        section = page.get_by_test_id('model-runs')
        section.wait_for(state='visible')
        response = page.request.get(args.url.rstrip('/') + '/api/report')
        assert response.status == 200
        report = response.json()
        matches = [r for r in report['model_runs'] if r['run_id'] == args.run_id]
        assert len(matches) == 1
        run = matches[0]
        assert run['extensions']['execution']['max_batch_size'] > 1
        # Attribute values are read rather than interpolated into CSS selectors.
        cards = section.locator('[data-run-id]')
        card = next(cards.nth(i) for i in range(cards.count()) if cards.nth(i).get_attribute('data-run-id') == args.run_id)
        assert card.is_visible()
        text = card.inner_text()
        assert args.run_id in text and 'optimizer' in text.lower() and 'prefix' in text.lower()
        assert not errors, errors
        page.screenshot(path=str(args.out/'overwatch-populated.png'), full_page=True)
        (args.out/'report.json').write_text(json.dumps({'url':args.url, 'run':run,
            'actual_report':report, 'console_errors':errors, 'card_text':text}, indent=2))
        browser.close()
if __name__ == '__main__': main()
