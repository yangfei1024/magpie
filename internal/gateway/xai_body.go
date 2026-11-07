package gateway

import (
	"bytes"
	"encoding/json"

	"github.com/yetone/magpie/internal/provider"
)

// xAI's own API (api.x.ai), wherever a Grok model is served, takes three
// things a client's wire puts in wrong, as the Grok subscription's plugin
// shapes its bodies too (provider's grokShaped):
//
//   - Codex hands a sealed reasoning item back with "content": null
//     beside its encrypted_content, which the backend answers "Could not
//     decode the compaction blob. Ensure it is unmodified from the compact
//     response." (#3392, kongkongklng: the first turn went through, every
//     one after it 400ed) — the null goes.
//   - Claude Code sends a system turn mid-conversation (Anthropic made
//     them GA), which xAI's Messages answers "Invalid request content:
//     Invalid message role." (#3392: every claude call 400ed) — each folds
//     into the top-level system, the only place it takes the role.
//   - A tool's schema can say "required": null where none are (Claude
//     Code's tools, an MCP server's among them — Anthropic's API takes
//     it) or leave the key out altogether (v0.1.2557's Claude Code sends
//     CronList and its kin so), which xAI answers "Invalid request
//     content: Schema validation failed: [standard_violation] /required:
//     null is not of type \"array\"" (#3392: the claude calls 400ed
//     after the system-turn fix, and again with none null in the body) —
//     every object node gets its required, an empty array where none
//     were.
//
// The relay and the translation shape alike: a refusal of custom tools
// puts a Codex request on the translated path from the next turn on
// (custom_tool.go), and there is nothing to learn from either answer —
// xAI takes no null content, no system turn, and every object schema's
// required.

// xaiBody shapes body as xAI's own API takes it on proto.
func xaiBody(p provider.Provider, proto provider.Protocol, body []byte) []byte {
	if p.Host() != "api.x.ai" {
		return body
	}
	body = toolsRequiredFilled(body)
	switch proto {
	case provider.Responses:
		return withoutNullReasoningContent(body)
	case provider.Anthropic:
		return systemTurnsFolded(body)
	}
	return body
}

// withoutNullReasoningContent is a Responses body with the null content of
// a reasoning item gone (#3392).
func withoutNullReasoningContent(body []byte) []byte {
	if !bytes.Contains(body, []byte(`"reasoning"`)) || !bytes.Contains(body, []byte(`"content"`)) {
		return body
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	var m map[string]any
	if dec.Decode(&m) != nil || m == nil {
		return body
	}
	dirty := false
	items, _ := m["input"].([]any)
	for _, it := range items {
		im, _ := it.(map[string]any)
		if im == nil || im["type"] != "reasoning" {
			continue
		}
		if c, there := im["content"]; there && c == nil {
			delete(im, "content")
			dirty = true
		}
	}
	if !dirty {
		return body
	}
	out, err := marshalPlain(m)
	if err != nil {
		return body
	}
	return out
}

// systemTurnsFolded is an Anthropic body with each system turn in messages
// folded into the top-level system, its text and blocks as they came
// (#3392).
func systemTurnsFolded(body []byte) []byte {
	if !bytes.Contains(body, []byte(`"messages"`)) || !bytes.Contains(body, []byte(`"system"`)) {
		return body
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	var m map[string]any
	if dec.Decode(&m) != nil || m == nil {
		return body
	}
	msgs, _ := m["messages"].([]any)
	kept := msgs[:0:0]
	var folded []any // each system turn's content, a string or blocks
	for _, mm := range msgs {
		im, _ := mm.(map[string]any)
		if im == nil || im["role"] != "system" {
			kept = append(kept, mm)
			continue
		}
		switch c := im["content"].(type) {
		case string:
			folded = append(folded, c)
		case []any:
			folded = append(folded, c...)
		}
	}
	if len(kept) == len(msgs) {
		return body
	}
	m["messages"] = kept
	// the system prompt: what was there, then each turn's text and blocks
	var blocks []any
	switch s := m["system"].(type) {
	case string:
		if s != "" {
			blocks = append(blocks, map[string]any{"type": "text", "text": s})
		}
	case []any:
		blocks = append(blocks, s...)
	}
	for _, f := range folded {
		if s, ok := f.(string); ok {
			if s != "" {
				blocks = append(blocks, map[string]any{"type": "text", "text": s})
			}
			continue
		}
		blocks = append(blocks, f)
	}
	if len(blocks) > 0 {
		m["system"] = blocks
	} else {
		delete(m, "system")
	}
	out, err := marshalPlain(m)
	if err != nil {
		return body
	}
	return out
}

// toolsRequiredFilled is a body whose tool schemas each say their
// required (#3392). xAI validates each tool's schema as a JSON Schema
// and refuses a required that is null — where a client's wire puts it
// when none are (Anthropic's API takes the null) — and takes one left
// out as that same null, as v0.1.2557's Claude Code showed: its tools
// that ask for nothing (CronList and its kin) leave the key out. Every
// object node therefore goes out with its required, an empty array
// where none were, the schemas found under every shape a body leaves
// magpie in: input_schema (Anthropic), parameters (Responses) and
// function's parameters (chat).
func toolsRequiredFilled(body []byte) []byte {
	if !bytes.Contains(body, []byte(`"tools"`)) {
		return body
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	var m map[string]any
	if dec.Decode(&m) != nil || m == nil {
		return body
	}
	dirty := false
	tools, _ := m["tools"].([]any)
	for _, t := range tools {
		tm, _ := t.(map[string]any)
		if tm == nil {
			continue
		}
		for _, k := range []string{"parameters", "input_schema"} {
			if fillRequired(tm[k]) {
				dirty = true
			}
		}
		if fm, _ := tm["function"].(map[string]any); fm != nil {
			if fillRequired(fm["parameters"]) {
				dirty = true
			}
		}
	}
	if !dirty {
		return body
	}
	out, err := marshalPlain(m)
	if err != nil {
		return body
	}
	return out
}

// fillRequired gives every object node of a schema its required — an
// array already there stays as it came — recursively through the
// schema's objects and its arrays alike, where a union's anyOf branch or
// a tuple items holds one, and says whether it touched anything. A null
// required on a node that is no object still goes: only an object node
// has the key mean anything, and xAI asks for it there.
func fillRequired(v any) bool {
	switch t := v.(type) {
	case map[string]any:
		dirty := false
		if t["type"] == "object" {
			if r, there := t["required"]; !there || r == nil {
				t["required"] = []any{}
				dirty = true
			}
		} else if r, there := t["required"]; there && r == nil {
			delete(t, "required")
			dirty = true
		}
		for _, child := range t {
			if fillRequired(child) {
				dirty = true
			}
		}
		return dirty
	case []any:
		dirty := false
		for _, child := range t {
			if fillRequired(child) {
				dirty = true
			}
		}
		return dirty
	}
	return false
}
