// Package cacheoptimizer provides request canonicalization to improve
// upstream cache hit rates for providers that key their cache on the raw
// request body bytes.
//
// DeepSeek uses a Prefix Cache that matches on exact token prefixes of the
// request. Every byte-level variation (key ordering, whitespace, tool call IDs,
// message ordering) can cause cache misses. This package canonicalizes chat
// completion requests through a 12-rule pipeline so that semantically identical
// inputs produce byte-identical outputs, maximizing DeepSeek's cache hit rate.
package optimizer

import (
	"bytes"
	"encoding/json"
	"strings"

	log "github.com/sirupsen/logrus"
)

// CanonicalizeDeepSeekRequest applies the full 12-rule canonicalization pipeline
// to a DeepSeek chat completion request body.
func canonicalizeDeepSeekRequest(body []byte) ([]byte, error) {
	if len(body) == 0 {
		return body, nil
	}

	var root map[string]any
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	if err := dec.Decode(&root); err != nil {
		return nil, err
	}

	// ---------------------------------------------------------------
	// Rule 4+8: Recursive key-sort of all map[string]any (includes tools schema).
	// Done FIRST so subsequent inspection sees stable keys.
	// ---------------------------------------------------------------
	canonicalizeAllKeys(root)

	// ---------------------------------------------------------------
	// Rule 8: Sort tools by function name.
	// ---------------------------------------------------------------
	if tools, ok := root["tools"]; ok {
		root["tools"] = sortToolsByName(tools)
	}

	// ---------------------------------------------------------------
	// Rules 1-3, 6-7, 9: Message-level canonicalization.
	// ---------------------------------------------------------------
	if msgs, ok := root["messages"]; ok {
		root["messages"] = canonicalizeMessagesDeep(msgs)
	}

	// ---------------------------------------------------------------
	// Rule 6 (top-level): Remove non-semantic top-level fields.
	// Removed: stream_options (response format, not prompt),
	// user (end-user identifier, varies per caller but is not part of prompt semantics).
	// ---------------------------------------------------------------
	delete(root, "stream_options")
	delete(root, "user")

	// ---------------------------------------------------------------
	// Final serialization: deterministic JSON with sorted keys.
	// ---------------------------------------------------------------
	output := encodeSorted(root)

	// ---------------------------------------------------------------
	// Rule 10: Whitespace normalization (on string content within JSON).
	// Apply after serialization by re-parsing, normalizing strings, re-encoding.
	// ---------------------------------------------------------------
	output = normalizeWhitespaceInPlace(output)

	// ---------------------------------------------------------------
	// Rule 11: Prompt fingerprint.
	// ---------------------------------------------------------------
	fingerprint := sha256sum(output)
	logDriftIfChanged(fingerprint, output)

	// ---------------------------------------------------------------
	// Rule 12: Cache simulation.
	// ---------------------------------------------------------------
	simulateCache(output)

	return output, nil
}

// isDeepSeekProvider checks whether the given provider identifier represents
// a DeepSeek-backed provider that should use cache optimization.
func isDeepSeekProvider(provider string) bool {
	provider = strings.ToLower(strings.TrimSpace(provider))
	return provider == "deepseek" || strings.HasPrefix(provider, "openai-compatible-deepseek")
}

// =============================================================================
// Public API
// =============================================================================

// DeepSeekApplyToRequest applies DeepSeek cache optimization to the translated request body.
func DeepSeekApplyToRequest(provider string, body []byte) []byte {
	if !isDeepSeekProvider(provider) {
		return body
	}
	canonicalized, err := canonicalizeDeepSeekRequest(body)
	if err != nil {
		log.WithError(err).Debug("deepseek cache optimizer: canonicalization failed, using original body")
		return body
	}
	return canonicalized
}
