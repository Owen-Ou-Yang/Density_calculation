# Security and private scientific data

Never post passwords, access tokens, SSH keys, private site configurations,
experimental tables or raw scientific results in public issues or pull requests.
Use a minimal synthetic reproducer and sanitized excerpts for ordinary bugs.

For a security or accidental-disclosure report, contact the maintainer privately
through an existing agreed channel. If no private channel is available, open an
issue asking how to contact the maintainer **without including the sensitive
details**. Do not assume private vulnerability reporting is enabled.

Removing a secret from the newest commit does not revoke it or erase history.
If credentials are exposed, revoke/rotate them at their provider and notify the
maintainer. `.gitignore` and the package checker reduce accidental inclusion;
they are not a guarantee that every confidential item will be detected.

