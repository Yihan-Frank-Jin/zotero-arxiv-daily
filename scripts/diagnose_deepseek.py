"""Probe DeepSeek without printing credentials or sending email."""
import json
import os
import socket
from urllib.parse import urlsplit

import httpx
import yaml
from openai import OpenAI


def report(label, **values):
    print(label, json.dumps(values, ensure_ascii=True), flush=True)


def failure(label, exc):
    # Malformed header/URL errors may contain secrets: log only exception types.
    chain, seen = [], set()
    current = exc
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        chain.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    report(label, ok=False, exception_types=chain, http_status=getattr(exc, 'status_code', None))


def main():
    base = os.environ.get('OPENAI_API_BASE', '')
    key = os.environ.get('OPENAI_API_KEY', '')
    report('CONFIG',
        base_present=bool(base),
        base_matches_official=base in {'https://api.deepseek.com', 'https://api.deepseek.com/', 'https://api.deepseek.com/v1', 'https://api.deepseek.com/v1/'},
        base_outer_whitespace=base != base.strip(),
        key_present=bool(key),
        key_outer_whitespace=key != key.strip(),
        key_contains_newline=any(c in key for c in '\r\n'),
        key_is_ascii=key.isascii(),
        proxy_environment_present=any(os.environ.get(k) for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')),
    )
    try:
        addresses = socket.getaddrinfo('api.deepseek.com', 443, type=socket.SOCK_STREAM)
        report('DNS', ok=bool(addresses))
    except Exception as exc:
        failure('DNS', exc)
    try:
        response = httpx.get('https://api.deepseek.com/models', timeout=20, follow_redirects=False)
        report('PUBLIC_HTTPS', http_status=response.status_code)
    except Exception as exc:
        failure('PUBLIC_HTTPS', exc)
    try:
        parts = urlsplit(base)
        if parts.scheme != 'https' or parts.hostname != 'api.deepseek.com' or parts.username or parts.password or parts.query or parts.fragment:
            report('AUTHENTICATED_PROBE', skipped=True, reason='Configured URL is not the expected official HTTPS endpoint')
            return
        if not key:
            report('AUTHENTICATED_PROBE', skipped=True, reason='API key is empty')
            return
        with OpenAI(api_key=key, base_url=base, max_retries=0, timeout=20) as client:
            try:
                models = client.models.list()
                report('MODELS', ok=True, model_count=len(models.data))
            except Exception as exc:
                failure('MODELS', exc)
                return
            config = yaml.safe_load(os.environ.get('CUSTOM_CONFIG', '')) or {}
            model = config.get('llm', {}).get('generation_kwargs', {}).get('model', 'deepseek-v4-flash')
            try:
                result = client.chat.completions.create(
                    model=model,
                    messages=[{'role': 'user', 'content': 'Reply OK.'}],
                    max_tokens=16,
                    extra_body={'thinking': {'type': 'disabled'}},
                )
                report('CHAT', ok=True, has_content=bool(result.choices and result.choices[0].message.content))
            except Exception as exc:
                failure('CHAT', exc)
    except Exception as exc:
        failure('CONFIGURED_CLIENT', exc)


if __name__ == '__main__':
    main()
