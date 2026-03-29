We are currently working on the main Slaze functionality.
The main entrypoint for the Slaze functionality is the run.py file.

## Known Gotchas
- Python/uv PATH is persisted in Windows Registry (HKCU\Environment). If `python` or `uv` not found in shell, refresh PATH: `$env:Path = [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + [Environment]::GetEnvironmentVariable('Path', 'Machine')`. May need to open new terminal or log off/on.
