#!/bin/sh
set -eu
PAPERLEET_DEPLOYMENT=oracle
COMPOSE_PROJECT_NAME=${COMPOSE_PROJECT_NAME:-paperleet-oracle}
export PAPERLEET_DEPLOYMENT COMPOSE_PROJECT_NAME
exec "$(dirname "$0")/manage.sh" "$@"
