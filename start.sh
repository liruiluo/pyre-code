#!/usr/bin/env bash
# Rebuild and restart the public pyre-code deployment.
#
# auth/htpasswd is gitignored; regenerate it with:
#   PASS='your-new-password'
#   printf 'Magipenguin:%s\n' "$(openssl passwd -apr1 "$PASS")" > auth/htpasswd
#   chmod 600 auth/htpasswd && echo "$PASS" > auth/password.txt
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f auth/htpasswd ]; then
    echo "auth/htpasswd missing — see regeneration instructions at the top of this script." >&2
    exit 1
fi
if [ ! -f auth/nginx.conf ]; then
    echo "auth/nginx.conf missing" >&2
    exit 1
fi

docker compose -f docker-compose.public.yml up -d --build
docker compose -f docker-compose.public.yml ps
