#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

# Docker deliberately omits a default route on --internal networks. The stock
# kind entrypoint assumes one exists while moving Docker's embedded DNS rules.
# Add only the on-link internal bridge gateway; Docker's internal-network
# forwarding policy remains the actual egress boundary and is smoke-tested.
# The harness may select a bounded fallback /24 after proving that the preferred
# subnet overlaps another Docker network, so derive .1 from the actual interface
# address rather than assuming the preferred gateway.
if ! ip -4 route show default | grep -q '^default '; then
  read -r gateway_device internal_address < <(
    ip -4 -o addr show scope global |
      awk '$4 ~ /\/24$/ { print $2, $4; exit }'
  )
  if [[ -z "${gateway_device:-}" || -z "${internal_address:-}" ]]; then
    echo "ERROR: could not find the internal Docker /24 interface" >&2
    exit 1
  fi
  if [[ ! "${internal_address}" =~ ^([0-9]+\.[0-9]+\.[0-9]+)\.[0-9]+/24$ ]]; then
    echo "ERROR: unexpected internal Docker address ${internal_address}" >&2
    exit 1
  fi
  internal_gateway="${BASH_REMATCH[1]}.1"
  if ! ip -4 route get "${internal_gateway}" | grep -Eq "dev ${gateway_device}( |$)"; then
    echo "ERROR: derived gateway ${internal_gateway} is not on ${gateway_device}" >&2
    exit 1
  fi
  ip -4 route add default via "${internal_gateway}" dev "${gateway_device}"
  echo "INFO: added internal-only default route via ${internal_gateway}" >&2
fi

exec /usr/local/bin/entrypoint "$@"
