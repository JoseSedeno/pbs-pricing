# PBS AEMP Viewer — Access Control Runbook

## Where to manage access
Streamlit → App → ⋮ (next to `pbs_viewer_app.py`) → **Edit secrets**

## Current baseline (example)
```toml
[auth]
AUTH_NONCE = "u8xPq9C2m7Wk4NZa"

[auth.PASSWORDS]
demo01 = "kmchealthcare-2025"
Matt_Kirchmann = "kmchealthcare-2025"
acme_trial = "Acme-2025-Trial!"

[auth.EXPIRES_UTC]
# Optional per-user expiry (omit users who should never expire)
acme_trial = "2025-12-31T23:59:59Z"

[drive]
DB_FILE_ID = "1tVpP0p3XdSPyzn_GEs6T_q7I1Zkk3Veb"
