"""The host hands binds each of its servers on: loopback's IPv4 address, which only processes on this machine reach.

[LAW:one-source-of-truth] the proxy, the MCP server, the brain's hooks' listener and the display route all listen here,
and the brain reaches them here directly, past its HTTPS proxy (hands.brain.process.environment).
"""

HOST = "127.0.0.1"
