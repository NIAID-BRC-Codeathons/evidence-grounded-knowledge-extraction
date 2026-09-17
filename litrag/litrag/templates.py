"""Server-driven data types.

Literature.js hardcoded its four data types and their column lists. The server
publishes exactly those through /v1/prompt-templates, versioned and hashed, so
this module fetches them instead. A template added server-side shows up in the
CLI and the web UI with no code change here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# Server template ids are awkward on a command line ("ppi-extraction"). These
# aliases are convenience only -- they never define which templates exist.
ALIASES = {
    "ast": "ast",
    "susceptibility": "ast",
    "amr": "ast",
    "mic": "ast",
    "ppi": "ppi-extraction",
    "protein_function": "protein-function",
    "protein-function": "protein-function",
    "function": "protein-function",
    "mutation": "mutation",
    "mutations": "mutation",
    "summary": "literature-summary",
    "literature_summary": "literature-summary",
    "none": "literature-summary",
}


class TemplateError(ValueError):
    """A template was unknown, or its slot values were invalid."""


@dataclass(frozen=True)
class Slot:
    name: str
    required: bool = False
    max_len: Optional[int] = None
    label: Optional[str] = None

    @classmethod
    def from_decl(cls, decl: Dict[str, Any]) -> "Slot":
        return cls(
            name=decl["name"],
            required=bool(decl.get("required", False)),
            max_len=decl.get("max_len"),
            label=decl.get("label"),
        )


@dataclass(frozen=True)
class Template:
    """A prompt template declaration.

    The server returns the declaration only -- id, version, hash, label, output
    shape, columns, slots -- never the prompt body. That is deliberate on the
    server's side, and it is why the "View / Edit Prompt" dialog in the original
    widget cannot be reproduced.
    """

    id: str
    label: str
    output: str = "text"
    version: Optional[int] = None
    hash: Optional[str] = None
    columns: Optional[List[str]] = None
    max_output_tokens: Optional[int] = None
    slots: List[Slot] = field(default_factory=list)
    # "server" for /v1/prompt-templates, "local" for one LitRAG defines.
    source: str = "server"
    # Extra prompt rules, used only on the local path where LitRAG writes the
    # prompt. The server's own templates carry their rules in their hidden body.
    guidance: List[str] = field(default_factory=list)

    @property
    def is_table(self) -> bool:
        return self.output == "table" and bool(self.columns)

    @property
    def is_local(self) -> bool:
        """True when the server does not know this template.

        Such a template can only run on a local generator, since the hosted
        /v1/query would reject the id.
        """
        return self.source == "local"

    @classmethod
    def from_decl(cls, decl: Dict[str, Any]) -> "Template":
        return cls(
            id=decl["id"],
            label=decl.get("label", decl["id"]),
            output=decl.get("output", "text"),
            version=decl.get("version"),
            hash=decl.get("hash"),
            columns=list(decl["columns"]) if decl.get("columns") else None,
            max_output_tokens=decl.get("max_output_tokens"),
            slots=[Slot.from_decl(s) for s in decl.get("slots", [])],
            source=decl.get("source", "server"),
            guidance=list(decl.get("guidance", [])),
        )

    def validate_vars(self, values: Dict[str, str]) -> Dict[str, str]:
        """Check slot values locally and return only the non-empty ones.

        Catching a missing required slot here turns a server 422 into an
        actionable message before any network call is made.
        """
        cleaned: Dict[str, str] = {}
        known = {slot.name for slot in self.slots}

        unknown = sorted(set(values) - known)
        if unknown:
            raise TemplateError(
                f"template '{self.id}' has no slot(s): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(known)) or 'none'}"
            )

        for slot in self.slots:
            raw = values.get(slot.name)
            value = (raw or "").strip()
            if not value:
                if slot.required:
                    label = slot.label or slot.name
                    raise TemplateError(
                        f"template '{self.id}' requires a value for '{slot.name}' ({label})"
                    )
                continue
            if slot.max_len and len(value) > slot.max_len:
                raise TemplateError(
                    f"'{slot.name}' is {len(value)} characters; "
                    f"template '{self.id}' allows at most {slot.max_len}"
                )
            cleaned[slot.name] = value
        return cleaned


class TemplateRegistry:
    """The templates this tenant offers, fetched once and cached."""

    def __init__(self, templates: Sequence[Template]) -> None:
        self._templates = {t.id: t for t in templates}

    @classmethod
    def from_declarations(cls, decls: Sequence[Dict[str, Any]]) -> "TemplateRegistry":
        return cls([Template.from_decl(d) for d in decls])

    @classmethod
    def fetch(cls, client: Any, include_local: bool = True) -> "TemplateRegistry":
        """Server templates, plus any LitRAG defines locally.

        A server template always wins a name clash: if the operator later adds
        a template with the same id, the hosted path becomes available and the
        local stand-in should step aside.
        """
        decls = list(client.prompt_templates())
        if include_local:
            from .local_templates import declarations
            known = {d.get("id") for d in decls}
            decls.extend(d for d in declarations() if d["id"] not in known)
        return cls.from_declarations(decls)

    def __len__(self) -> int:
        return len(self._templates)

    def __iter__(self):
        return iter(self._templates.values())

    @property
    def ids(self) -> List[str]:
        return sorted(self._templates)

    def resolve(self, name: str) -> Template:
        """Look up a template by id or alias."""
        if not name:
            raise TemplateError("no data type given")
        key = name.strip()
        candidate = self._templates.get(key)
        if candidate:
            return candidate

        aliased = ALIASES.get(key.lower().replace(" ", "_"))
        if aliased and aliased in self._templates:
            return self._templates[aliased]

        lowered = {t.lower(): t for t in self._templates}
        if key.lower() in lowered:
            return self._templates[lowered[key.lower()]]

        raise TemplateError(
            f"unknown data type '{name}'. Available: {', '.join(self.ids)}"
        )
