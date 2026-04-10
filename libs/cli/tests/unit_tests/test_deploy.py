"""Tests for deploy bundler provider coverage.

Ensures that every model provider passing credential validation also gets
its partner package bundled into the deployment ``pyproject.toml``. Without
this, ``deepagents deploy`` succeeds but the deployed agent crashes with
``ImportError`` at runtime.
"""

from __future__ import annotations

import json
import tomllib
from typing import TYPE_CHECKING

import pytest

from deepagents_cli.deploy.bundler import (
    _MODEL_PROVIDER_DEPS,
    _render_deploy_graph,
    _render_pyproject,
    bundle,
)
from deepagents_cli.deploy.config import (
    _MODEL_PROVIDER_ENV,
    _parse_config,
    _validate_model_credentials,
    extract_provider,
)
from deepagents_cli.deploy.templates import SANDBOX_BLOCKS

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Dict-level invariant tests
# ---------------------------------------------------------------------------


class TestProviderDictSync:
    """The two provider dicts must stay in lockstep."""

    def test_every_validated_provider_has_a_bundler_dep(self) -> None:
        """Providers missing from bundler crash at deploy time.

        Any provider in ``_MODEL_PROVIDER_ENV`` that is missing from
        ``_MODEL_PROVIDER_DEPS`` will pass credential validation but
        cause ``ImportError`` at runtime.
        """
        missing = set(_MODEL_PROVIDER_ENV) - set(_MODEL_PROVIDER_DEPS)
        assert not missing, (
            f"Providers validated in config but missing from bundler deps: "
            f"{sorted(missing)}"
        )

    def test_no_bundler_dep_without_validated_provider(self) -> None:
        """Providers in bundler but not in config are dead deps."""
        extra = set(_MODEL_PROVIDER_DEPS) - set(_MODEL_PROVIDER_ENV)
        assert not extra, (
            f"Providers in bundler deps but not validated in config: {sorted(extra)}"
        )


class TestExtractProvider:
    """``extract_provider`` normalizes the provider prefix consistently."""

    def test_extracts_lowercase_provider(self) -> None:
        assert extract_provider("openai:gpt-4o") == "openai"

    def test_normalizes_uppercase(self) -> None:
        assert extract_provider("OPENAI:gpt-4o") == "openai"

    def test_strips_whitespace(self) -> None:
        assert extract_provider(" openai :gpt-4o") == "openai"

    def test_no_colon_returns_empty(self) -> None:
        assert extract_provider("gpt-4o") == ""

    def test_empty_string_returns_empty(self) -> None:
        assert extract_provider("") == ""


# ---------------------------------------------------------------------------
# Package-name spot checks -- non-obvious mappings that break silently
# ---------------------------------------------------------------------------


class TestNonObviousPackageMappings:
    """Guard provider-to-package mappings that don't follow the naming convention."""

    def test_nvidia_maps_to_ai_endpoints_package(self) -> None:
        assert _MODEL_PROVIDER_DEPS["nvidia"] == "langchain-nvidia-ai-endpoints"

    def test_azure_openai_shares_openai_package(self) -> None:
        assert _MODEL_PROVIDER_DEPS["azure_openai"] == "langchain-openai"

    def test_google_vertexai_maps_to_correct_package(self) -> None:
        assert _MODEL_PROVIDER_DEPS["google_vertexai"] == "langchain-google-vertexai"

    def test_google_genai_maps_to_correct_package(self) -> None:
        assert _MODEL_PROVIDER_DEPS["google_genai"] == "langchain-google-genai"

    def test_bedrock_maps_to_aws_package(self) -> None:
        assert _MODEL_PROVIDER_DEPS["bedrock"] == "langchain-aws"

    def test_perplexity_env_var_is_pplx(self) -> None:
        """``PPLX_API_KEY`` is non-obvious but correct for langchain-perplexity."""
        assert _MODEL_PROVIDER_ENV["perplexity"] == "PPLX_API_KEY"

    def test_bedrock_accepts_multiple_auth_methods(self) -> None:
        """Bedrock supports access keys, profiles, and IAM roles."""
        env_vars = _MODEL_PROVIDER_ENV["bedrock"]
        assert isinstance(env_vars, list)
        assert "AWS_ACCESS_KEY_ID" in env_vars
        assert "AWS_PROFILE" in env_vars
        assert "AWS_ROLE_ARN" in env_vars


# ---------------------------------------------------------------------------
# _render_pyproject -- the actual code path where the bug manifests
# ---------------------------------------------------------------------------


class TestRenderPyproject:
    """Verify ``_render_pyproject`` emits the correct dep for each provider."""

    @pytest.mark.parametrize(
        ("provider", "expected_pkg"),
        sorted(_MODEL_PROVIDER_DEPS.items()),
        ids=sorted(_MODEL_PROVIDER_DEPS),
    )
    def test_provider_dep_appears_in_rendered_pyproject(
        self,
        provider: str,
        expected_pkg: str,
    ) -> None:
        config = _parse_config(
            {"agent": {"name": "t", "model": f"{provider}:test-model"}},
        )
        result = _render_pyproject(config, mcp_present=False)
        assert expected_pkg in result, (
            f"_render_pyproject did not include {expected_pkg!r} "
            f"for model {provider}:test-model"
        )

    def test_model_without_colon_adds_no_provider_dep(self) -> None:
        config = _parse_config({"agent": {"name": "t", "model": "gpt-4o"}})
        result = _render_pyproject(config, mcp_present=False)
        for pkg in _MODEL_PROVIDER_DEPS.values():
            assert pkg not in result

    def test_unknown_provider_adds_no_dep(self) -> None:
        config = _parse_config(
            {"agent": {"name": "t", "model": "unknown_provider:v1"}},
        )
        result = _render_pyproject(config, mcp_present=False)
        for pkg in _MODEL_PROVIDER_DEPS.values():
            assert pkg not in result

    def test_unknown_provider_fails_validation(self) -> None:
        errors = _validate_model_credentials("unknown_provider:v1")
        assert len(errors) == 1
        assert "unknown_provider" in errors[0]
        assert "Valid providers" in errors[0]

    def test_mcp_present_adds_mcp_adapters(self) -> None:
        config = _parse_config(
            {"agent": {"name": "t", "model": "fireworks:llama"}},
        )
        result = _render_pyproject(config, mcp_present=True)
        assert "langchain-mcp-adapters" in result
        assert "langchain-fireworks" in result

    def test_mcp_absent_excludes_mcp_adapters(self) -> None:
        config = _parse_config(
            {"agent": {"name": "t", "model": "openai:gpt-4o"}},
        )
        result = _render_pyproject(config, mcp_present=False)
        assert "langchain-mcp-adapters" not in result

    def test_sandbox_provider_adds_partner_package(self) -> None:
        config = _parse_config(
            {
                "agent": {"name": "t", "model": "nvidia:nemotron"},
                "sandbox": {"provider": "daytona"},
            },
        )
        result = _render_pyproject(config, mcp_present=False)
        assert "langchain-nvidia-ai-endpoints" in result
        assert "langchain-daytona" in result

    def test_uppercase_provider_is_normalized(self) -> None:
        config = _parse_config(
            {"agent": {"name": "t", "model": "OPENAI:gpt-4o"}},
        )
        result = _render_pyproject(config, mcp_present=False)
        assert "langchain-openai" in result

    def test_whitespace_provider_is_normalized(self) -> None:
        config = _parse_config(
            {"agent": {"name": "t", "model": " openai :gpt-4o"}},
        )
        result = _render_pyproject(config, mcp_present=False)
        assert "langchain-openai" in result

    def test_rendered_pyproject_is_valid_toml(self) -> None:
        config = _parse_config(
            {"agent": {"name": "t", "model": "anthropic:claude"}},
        )
        result = _render_pyproject(config, mcp_present=True)
        parsed = tomllib.loads(result)
        assert parsed["project"]["name"] == "t"


# ---------------------------------------------------------------------------
# Full bundle() -- end-to-end through the real user path
# ---------------------------------------------------------------------------


class TestBundleEndToEnd:
    """Run ``bundle()`` the way ``deepagents deploy`` does."""

    def test_bundle_generates_all_required_files(self, tmp_path: Path) -> None:
        proj = tmp_path / "project"
        proj.mkdir()
        (proj / "AGENTS.md").write_text("You are a test agent.")

        config = _parse_config(
            {"agent": {"name": "t", "model": "xai:grok"}},
        )
        build = tmp_path / "build"
        bundle(config, proj, build)

        assert (build / "pyproject.toml").is_file()
        assert (build / "deploy_graph.py").is_file()
        assert (build / "langgraph.json").is_file()
        assert (build / "_seed.json").is_file()

    def test_bundle_seeds_agents_md(self, tmp_path: Path) -> None:
        proj = tmp_path / "project"
        proj.mkdir()
        (proj / "AGENTS.md").write_text("Custom system prompt here.")

        config = _parse_config(
            {"agent": {"name": "t", "model": "deepseek:chat"}},
        )
        build = tmp_path / "build"
        bundle(config, proj, build)

        seed = json.loads((build / "_seed.json").read_text(encoding="utf-8"))
        assert "/AGENTS.md" in seed["memories"]
        assert seed["memories"]["/AGENTS.md"] == "Custom system prompt here."

    def test_bundle_copies_mcp_json(self, tmp_path: Path) -> None:
        proj = tmp_path / "project"
        proj.mkdir()
        (proj / "AGENTS.md").write_text("Agent.")
        mcp_data = {
            "mcpServers": {"s": {"type": "http", "url": "http://localhost"}},
        }
        (proj / "mcp.json").write_text(json.dumps(mcp_data))

        config = _parse_config(
            {"agent": {"name": "t", "model": "cohere:command-r"}},
        )
        build = tmp_path / "build"
        bundle(config, proj, build)

        assert (build / "_mcp.json").is_file()
        pyproject = (build / "pyproject.toml").read_text(encoding="utf-8")
        assert "langchain-mcp-adapters" in pyproject
        assert "langchain-cohere" in pyproject

    def test_bundle_includes_skills(self, tmp_path: Path) -> None:
        proj = tmp_path / "project"
        proj.mkdir()
        (proj / "AGENTS.md").write_text("Agent.")
        skill_dir = proj / "skills" / "review"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("Review code.")

        config = _parse_config(
            {"agent": {"name": "t", "model": "openrouter:qwen"}},
        )
        build = tmp_path / "build"
        bundle(config, proj, build)

        seed = json.loads((build / "_seed.json").read_text(encoding="utf-8"))
        assert "/review/SKILL.md" in seed["skills"]
        assert seed["skills"]["/review/SKILL.md"] == "Review code."

        pyproject = (build / "pyproject.toml").read_text(encoding="utf-8")
        assert "langchain-openrouter" in pyproject

    def test_bundle_copies_env_file(self, tmp_path: Path) -> None:
        proj = tmp_path / "project"
        proj.mkdir()
        (proj / "AGENTS.md").write_text("Agent.")
        (proj / ".env").write_text("PPLX_API_KEY=xxx")

        config = _parse_config(
            {"agent": {"name": "t", "model": "perplexity:sonar"}},
        )
        build = tmp_path / "build"
        bundle(config, proj, build)

        assert (build / ".env").is_file()
        lg = json.loads((build / "langgraph.json").read_text(encoding="utf-8"))
        assert lg.get("env") == ".env"

        pyproject = (build / "pyproject.toml").read_text(encoding="utf-8")
        assert "langchain-perplexity" in pyproject

    def test_bundle_model_in_deploy_graph(self, tmp_path: Path) -> None:
        proj = tmp_path / "project"
        proj.mkdir()
        (proj / "AGENTS.md").write_text("Agent.")

        config = _parse_config(
            {"agent": {"name": "t", "model": "baseten:llama-3"}},
        )
        build = tmp_path / "build"
        bundle(config, proj, build)

        graph = (build / "deploy_graph.py").read_text(encoding="utf-8")
        assert "baseten:llama-3" in graph


# ---------------------------------------------------------------------------
# Syntax validation -- generated deploy_graph.py must compile
# ---------------------------------------------------------------------------

_SANDBOX_MCP_PAIRS = [
    (sandbox, mcp) for sandbox in sorted(SANDBOX_BLOCKS) for mcp in (False, True)
]


class TestGeneratedArtifactValidity:
    """Generated deploy_graph.py must compile as valid Python."""

    @pytest.mark.parametrize(
        ("sandbox", "mcp_present"),
        _SANDBOX_MCP_PAIRS,
        ids=[f"{s}-mcp-{'on' if m else 'off'}" for s, m in _SANDBOX_MCP_PAIRS],
    )
    def test_deploy_graph_compiles_as_valid_python(
        self,
        sandbox: str,
        mcp_present: bool,
    ) -> None:
        cfg_dict: dict = {"agent": {"name": "t", "model": "anthropic:v1"}}
        if sandbox != "none":
            cfg_dict["sandbox"] = {"provider": sandbox}
        config = _parse_config(cfg_dict)
        code = _render_deploy_graph(
            config,
            "You are a test agent.",
            mcp_present=mcp_present,
        )
        compile(code, f"deploy_graph_{sandbox}_mcp{mcp_present}.py", "exec")
