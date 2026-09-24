#!/usr/bin/env python3
"""Generic read-only GitHub repository review agent."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import sys
from dataclasses import dataclass

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware import ModelCallLimitMiddleware, TodoListMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama

DEFAULT_OLLAMA_MODEL = "muse-glimmer:30b-mlx"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"

_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class ReviewConfig:
    repository: str
    model_provider: str
    ollama_model: str
    anthropic_model: str
    summarize_at_tokens: int
    keep_recent_tokens: int
    summary_input_tokens: int
    max_model_calls: int
    recursion_limit: int
    show_tools: bool
    mcp_mode: str
    mcp_binary: str


def parse_args() -> ReviewConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Review the health of any GitHub repository with a read-only "
            "Deep Agent backed by GitHub MCP."
        )
    )
    parser.add_argument("repository", help="GitHub repository in OWNER/REPO form.")
    parser.add_argument(
        "--provider",
        choices=("ollama", "anthropic"),
        default=os.getenv("MODEL_PROVIDER", "ollama").lower(),
        help="Model provider (default: MODEL_PROVIDER or ollama).",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        help=f"Ollama model (default: {DEFAULT_OLLAMA_MODEL}).",
    )
    parser.add_argument(
        "--anthropic-model",
        default=os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
        help=f"Anthropic model (default: {DEFAULT_ANTHROPIC_MODEL}).",
    )
    parser.add_argument(
        "--summarize-at-tokens",
        type=int,
        default=int(os.getenv("REVIEW_SUMMARIZE_AT_TOKENS", "55000")),
    )
    parser.add_argument(
        "--keep-recent-tokens",
        type=int,
        default=int(os.getenv("REVIEW_KEEP_RECENT_TOKENS", "12000")),
    )
    parser.add_argument(
        "--summary-input-tokens",
        type=int,
        default=int(os.getenv("REVIEW_SUMMARY_INPUT_TOKENS", "20000")),
    )
    parser.add_argument(
        "--max-model-calls",
        type=int,
        default=int(os.getenv("REVIEW_MAX_MODEL_CALLS", "30")),
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=int(os.getenv("REVIEW_RECURSION_LIMIT", "100")),
    )
    parser.add_argument(
        "--show-tools",
        action="store_true",
        help="Print loaded GitHub MCP tool names before starting.",
    )
    parser.add_argument(
        "--mcp-mode",
        choices=("auto", "docker", "native"),
        default=os.getenv("GITHUB_MCP_MODE", "auto").lower(),
        help=(
            "How to launch GitHub MCP. 'auto' uses a native "
            "github-mcp-server when available, otherwise Docker."
        ),
    )
    parser.add_argument(
        "--mcp-binary",
        default=os.getenv("GITHUB_MCP_BINARY", "github-mcp-server"),
        help="Native GitHub MCP executable name/path.",
    )

    args = parser.parse_args()

    if not _REPOSITORY_PATTERN.fullmatch(args.repository):
        parser.error("repository must be in OWNER/REPO form")

    positive_values = {
        "--summarize-at-tokens": args.summarize_at_tokens,
        "--keep-recent-tokens": args.keep_recent_tokens,
        "--summary-input-tokens": args.summary_input_tokens,
        "--max-model-calls": args.max_model_calls,
        "--recursion-limit": args.recursion_limit,
    }
    for flag, value in positive_values.items():
        if value <= 0:
            parser.error(f"{flag} must be positive")

    if args.keep_recent_tokens >= args.summarize_at_tokens:
        parser.error("--keep-recent-tokens must be smaller than --summarize-at-tokens")

    return ReviewConfig(
        repository=args.repository,
        model_provider=args.provider,
        ollama_model=args.ollama_model,
        anthropic_model=args.anthropic_model,
        summarize_at_tokens=args.summarize_at_tokens,
        keep_recent_tokens=args.keep_recent_tokens,
        summary_input_tokens=args.summary_input_tokens,
        max_model_calls=args.max_model_calls,
        recursion_limit=args.recursion_limit,
        show_tools=args.show_tools,
        mcp_mode=args.mcp_mode,
        mcp_binary=args.mcp_binary,
    )


def make_model(config: ReviewConfig):
    if config.model_provider == "ollama":
        return ChatOllama(
            model=config.ollama_model,
            temperature=0,
            validate_model_on_init=True,
        )

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "MODEL_PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set."
        )

    return ChatAnthropic(model=config.anthropic_model)


def make_github_client(
    token: str,
    config: ReviewConfig,
) -> tuple[MultiServerMCPClient, str]:
    """Create a read-only GitHub MCP client.

    Native mode avoids Docker when github-mcp-server is installed locally.
    Auto mode prefers the native binary and falls back to Docker.
    """

    mode = config.mcp_mode
    if mode == "auto":
        mode = "native" if shutil.which(config.mcp_binary) else "docker"

    common_env = {
        "GITHUB_PERSONAL_ACCESS_TOKEN": token,
        "GITHUB_READ_ONLY": "1",
        "GITHUB_TOOLSETS": "repos,issues,pull_requests,actions",
    }

    if mode == "native":
        binary = shutil.which(config.mcp_binary) or config.mcp_binary
        connection = {
            "transport": "stdio",
            "command": binary,
            "args": [
                "stdio",
                "--toolsets=repos,issues,pull_requests,actions",
                "--read-only",
            ],
            "env": common_env,
        }
    else:
        connection = {
            "transport": "stdio",
            "command": "docker",
            "args": [
                "run",
                "-i",
                "--rm",
                "-e",
                "GITHUB_PERSONAL_ACCESS_TOKEN",
                "-e",
                "GITHUB_READ_ONLY",
                "-e",
                "GITHUB_TOOLSETS",
                "ghcr.io/github/github-mcp-server:latest",
            ],
            "env": common_env,
        }

    client = MultiServerMCPClient(
        {"github": connection},
        tool_name_prefix=True,
        handle_tool_errors=True,
    )
    return client, mode


def make_system_prompt(repository: str) -> str:
    return f"""
You are a read-only maintenance-review agent for:

    {repository}

Your purpose is to identify repository work that warrants human attention.

Use GitHub tools to gather current evidence. Do not rely on prior knowledge.

Use planning and subagents when they help keep independent investigations
separate. You may use your scratch filesystem for concise intermediate notes.

========================================================================
REVIEW AREAS
========================================================================

1. CI HEALTH

Inspect recent GitHub Actions activity on the default branch.

Determine:
- whether the default branch is healthy;
- which workflows/jobs are failing;
- whether failures are recent or persistent;
- likely causes of meaningful failures.

Distinguish where evidence permits among:
- code regressions;
- flaky tests;
- compiler/platform compatibility problems;
- dependency failures;
- CI configuration problems;
- infrastructure failures.

Investigate progressively:
- start from workflow/job metadata;
- fetch logs only for relevant failed jobs;
- prefer a small log tail when the GitHub tool supports it;
- stop once evidence is sufficient.

2. PULL REQUESTS

Review currently open pull requests.

For PRs warranting attention determine:
- purpose;
- CI state;
- review/blocker state;
- staleness;
- dependencies or overlap with other work;
- recommended next human action.

Do not spend substantial context on routine healthy PRs.

3. ISSUES

Review currently open issues.

Pay particular attention to:
- regressions and confirmed bugs;
- CI/build problems;
- blockers;
- active or overlapping PRs;
- stale work;
- straightforward-fix candidates.

A straightforward-fix candidate requires evidence that the change is localized
and technically clear. Do not infer simplicity from a short issue description.

4. REPOSITORY HEALTH

Identify other evidence-supported maintenance concerns, such as:
- persistent CI failures;
- stale maintenance PRs;
- packaging/dependency problems;
- testing gaps;
- documentation maintenance;
- duplicated or abandoned work.

Do not perform a broad source audit unless a specific CI failure, issue, or PR
requires source inspection.

========================================================================
CONTEXT DISCIPLINE
========================================================================

GitHub tool responses can be large.

- Start with small result pages (roughly 10-20 items).
- Paginate only when needed.
- Do not query information already obtained.
- Do not repeatedly inspect the same object.
- Avoid complete CI logs and large source files unless necessary.
- Stop investigating once evidence is sufficient.
- Use scratch files for concise intermediate notes.
- Delegate independent CI/PR/issue investigations when useful.

Aim to complete the review with a bounded number of GitHub calls rather than
building an exhaustive mirror of the repository.

========================================================================
AUTHORITY
========================================================================

You are READ ONLY.

Never:
- create/edit/comment on issues;
- create branches or commits;
- modify repository contents;
- open/comment on/merge pull requests;
- alter Actions workflows or repository settings.

Recommend actions rather than performing them.

========================================================================
FINAL REPORT
========================================================================

Return:

# Repository review: {repository}

## Executive summary
Concise assessment of current repository health.

## CI
For each meaningful concern:
- workflow/job;
- status;
- likely cause;
- evidence;
- recommended next step.

If CI appears healthy, say so.

## Pull requests
For PRs warranting attention:
- number/title;
- current state;
- blocker or concern;
- recommended next action.

## Issues
Organize meaningful issues under:
### High attention
### Normal maintenance
### Straightforward fix candidates
### Stale / low priority

Do not force an issue into a category when evidence is insufficient.

## Recommended priorities
Provide a short ordered list of maintenance actions that deserve attention next.
""".strip()


def make_review_task(repository: str) -> str:
    return f"""
Perform a current maintenance review of {repository}.

Use GitHub tools to:
1. assess Actions health on the default branch;
2. investigate meaningful recent CI failures;
3. review open pull requests;
4. review open issues;
5. identify evidence-supported straightforward-fix candidates;
6. identify other repository-maintenance concerns.

Do not modify anything. Return the report specified in your instructions.
""".strip()


def extract_final_text(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "Agent returned no messages."

    content = messages[-1].content
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
        if parts:
            return "\n".join(parts)

    return str(content)


async def main() -> None:
    config = parse_args()

    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not github_token:
        print("ERROR: GITHUB_PERSONAL_ACCESS_TOKEN is not set.", file=sys.stderr)
        raise SystemExit(2)

    model = make_model(config)
    backend = StateBackend()

    summarization = SummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=("tokens", config.summarize_at_tokens),
        keep=("tokens", config.keep_recent_tokens),
        trim_tokens_to_summarize=config.summary_input_tokens,
    )

    github_client, mcp_mode = make_github_client(github_token, config)

    model_name = (
        config.ollama_model
        if config.model_provider == "ollama"
        else config.anthropic_model
    )

    print("=" * 78)
    print("Generic repository review Deep Agent")
    print("=" * 78)
    print(f"Repository : {config.repository}")
    print(f"Provider   : {config.model_provider}")
    print(f"Model      : {model_name}")
    print(f"GitHub MCP : {mcp_mode} / read only")
    print("Shell      : disabled")
    print()

    async with github_client.session("github") as github_session:
        github_tools = await load_mcp_tools(
            github_session,
            callbacks=github_client.callbacks,
            tool_interceptors=github_client.tool_interceptors,
            server_name="github",
            tool_name_prefix=github_client.tool_name_prefix,
            handle_tool_errors=github_client.handle_tool_errors,
        )

        print(f"Loaded {len(github_tools)} GitHub tools.")
        if config.show_tools:
            for tool in github_tools:
                print(f"  - {tool.name}")
        print()

        agent = create_deep_agent(
            name="repository-reviewer",
            model=model,
            tools=github_tools,
            system_prompt=make_system_prompt(config.repository),
            backend=backend,
            middleware=[
                summarization,
                TodoListMiddleware(),
                ModelCallLimitMiddleware(
                    run_limit=config.max_model_calls,
                    exit_behavior="end",
                ),
            ],
        )

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": make_review_task(config.repository),
                    }
                ]
            },
            config={"recursion_limit": config.recursion_limit},
        )

    print("=" * 78)
    print("REPOSITORY REVIEW")
    print("=" * 78)
    print(extract_final_text(result))


if __name__ == "__main__":
    asyncio.run(main())
