# Attention

## WSL2 NCCL MTU

ShardGrid WSL2 cross-host NCCL currently assumes:

- NCCL path interface MTU: `1500`
- target PMTU: `1500`
- no fixed TCP MSS override

Do not hard-code `eth0` / `eth1` / `eth3`.

Always resolve the live egress interface from the peer IP:

```bash
ip route get <peer_ip>
ip link show <dev>
```

Expected DF ping boundary for IPv4 MTU 1500:

```bash
ping -M do -s 1472 <peer_ip>  # should pass
ping -M do -s 1473 <peer_ip>  # should fail or be blocked
```

If NCCL symptoms look like:

- `dist.send()` returned
- `dist.recv()` returned
- receiver-side CUDA/NCCL completion hangs
- `ss -tinp` shows `Send-Q` buildup, retrans, ACK stall, or oversized `pmtu`

check MTU / PMTU mismatch first.

Do not try to hide this with:

- `sleep()`
- extra ACK messages
- fixed TCP MSS
- disabling PMTU discovery
- switching to Gloo just to make the test pass
- shrinking tensor sizes instead of fixing the path
