FROM kindest/node:v1.35.5@sha256:ce977ae6d65918d0b58a5f8b5e940429c2ce42fa3a5619ec2bbc60b949c0ac95

COPY benchmark/environment/kind-internal-entrypoint.sh /usr/local/bin/aipycraft-internal-entrypoint
RUN chmod 0755 /usr/local/bin/aipycraft-internal-entrypoint

ENTRYPOINT ["/usr/local/bin/aipycraft-internal-entrypoint", "/sbin/init"]
