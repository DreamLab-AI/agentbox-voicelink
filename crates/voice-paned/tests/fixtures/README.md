# Fixtures

`real_*` files are genuine `tmux capture-pane` output (`.txt` = `-p`, `.ansi` =
`-e -p`) from Claude Code 2.1.306 and Codex 0.163.0, captured in a private
tmux server. Everything else (`claude_*.txt`, `codex_*.txt`) is hand-built from
the CLIs' known menu layouts because a real approval menu needs a live model
turn.

Refresh these against real captures whenever either CLI is upgraded, and adjust
the tables in `src/prompt.rs` and the heuristics in `src/input.rs` to match.
