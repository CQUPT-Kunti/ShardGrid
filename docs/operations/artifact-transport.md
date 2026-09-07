# Artifact Transport

T087 defines one artifact transport contract over mature system tools only.
ShardGrid does not implement a custom file transfer protocol.

Supported transports:

- `rsync`
- `scp`
- `sftp`

Selection:

- explicit config uses the requested transport and fails if its executable is unavailable
- `auto` selects the first available transport in this order: `rsync`, `scp`, `sftp`
- missing tools are reported as unavailable; nothing is auto-installed

Failure semantics:

- permission failures stay failed and keep the tool, source, destination, exit code, and stderr
- multi-artifact runs can return `partial` when some items succeed and some fail
- partial runs are never reported as full success

Credential safety:

- passwords and tokens are never added to transport commands
- recorded commands and stderr go through the existing redaction path
- private key contents are never copied into logs or results

Scope boundary:

- T087 only provides transport abstraction, selection, and failure semantics
- real remote snapshot distribution starts in T088
