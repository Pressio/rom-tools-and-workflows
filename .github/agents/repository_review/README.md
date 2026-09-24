# Generic repository review agent

Read-only GitHub repository health reviewer built with LangChain Deep Agents and GitHub's official MCP server. The target repository is selected at runtime.

It reviews GitHub Actions, open pull requests, open issues, evidence-supported straightforward-fix candidates, and broader maintenance concerns. The MCP server is launched in read-only mode, so the agent cannot modify GitHub.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r .github/agents/repository_review/requirements.txt
```

Set a GitHub token with read access to the target repository:

```bash
export GITHUB_PERSONAL_ACCESS_TOKEN="github_pat_..."
```

## Ollama / Muse

Muse is the default backend:

```bash
export MODEL_PROVIDER=ollama
export OLLAMA_MODEL=muse-glimmer:30b-mlx

python .github/agents/repository_review/review_repository.py \
    Pressio/rom-tools-and-workflows
```

## Anthropic

```bash
export MODEL_PROVIDER=anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
export ANTHROPIC_MODEL=claude-sonnet-5

python .github/agents/repository_review/review_repository.py \
    Pressio/rom-tools-and-workflows
```

The provider can also be selected with `--provider ollama` or `--provider anthropic`.

## GitHub MCP launch mode

The default `--mcp-mode auto` uses a native `github-mcp-server` when one is available on `PATH`; otherwise it falls back to the official Docker image. Explicit modes are:

- `--mcp-mode native`
- `--mcp-mode docker`

A custom native binary can be selected with `GITHUB_MCP_BINARY` or `--mcp-binary`.

The server is restricted to the `repos`, `issues`, `pull_requests`, and `actions` toolsets and runs with GitHub MCP read-only mode enabled.

## Context controls

Repository reviews can accumulate large GitHub responses. The agent uses Deep Agents conversation compaction and conservative limits. Defaults are:

| Setting | Default |
| --- | ---: |
| summarize after | 55,000 message tokens |
| recent context retained | 12,000 tokens |
| history passed to summarizer | 20,000 tokens |
| model calls per review | 30 |
| LangGraph recursion limit | 100 |

These can be changed with CLI flags or the `REVIEW_SUMMARIZE_AT_TOKENS`, `REVIEW_KEEP_RECENT_TOKENS`, `REVIEW_SUMMARY_INPUT_TOKENS`, `REVIEW_MAX_MODEL_CALLS`, and `REVIEW_RECURSION_LIMIT` environment variables.

## Safety boundary

This tool is intentionally a reviewer, not a fixer. It may recommend creating issues, branches, or pull requests, but GitHub MCP read-only mode removes write tools at the MCP boundary. A future coding/fix agent should use a separate execution environment and separate GitHub write permissions rather than weakening this reviewer's read-only boundary.
