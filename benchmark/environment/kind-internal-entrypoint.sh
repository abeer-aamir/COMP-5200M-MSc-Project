#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

# Docker deliberately omits a default route on --internal networks. The stock
# kind entrypoint assumes one exists while moving Docker's embedded DNS rules.
# Add only the on-link internal bridge gateway; Docker's internal-network
# forwarding policy remains the actual egress boundary and is smoke-tested.
internal_gateway="${AIPYCRAFT_INTERNAL_GATEWAY:-172.31.250.1}"
if ! ip -4 route show default | grep -q '^default '; then
  gateway_device="$(
    ip -4 route get "${internal_gateway}" |
      awk '{ for (i = 1; i <= NF; i++) if ($i == "dev") { print $(i + 1); exit } }'
  )"
  if [[ -z "${gateway_device}" ]]; then
    echo "ERROR: internal Docker gateway ${internal_gateway} is not on-link" >&2
    exit 1
  fi
  ip -4 route add default via "${internal_gateway}" dev "${gateway_device}"
  echo "INFO: added internal-only default route via ${internal_gateway}" >&2
fi

exec /usr/local/bin/entrypoint "$@"
