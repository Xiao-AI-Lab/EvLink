# Security Policy

## Supported Versions

Security fixes are applied to the latest released EvLink version. The
project is currently pre-1.0, so users should review release notes before
upgrading across minor versions.

## Reporting a Vulnerability

Do not open a public issue for an unpatched vulnerability or exposed
credential. Use GitHub's private vulnerability reporting for the
`Xiao-AI-Lab/EvLink` repository. Include the affected version, reproduction
steps, impact, and any suggested mitigation.

## Secrets and External Services

EvLink accepts OpenAI-compatible endpoint URLs and API keys at runtime.
Keep keys in environment variables or a secret manager. Never commit them to
configuration files, artifacts, examples, logs, or caches. The default public
examples and CI do not call external model services.

Downloaded datasets and third-party retriever outputs are untrusted inputs.
Verify the checksum manifest, review upstream licenses, and avoid loading
untrusted pickle or database files.
