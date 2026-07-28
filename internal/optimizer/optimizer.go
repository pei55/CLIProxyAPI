/**
*@Author: pei5
*@Date: 2026/7/15 13:32
*@File: internal/optimizer/optimizer.go
*@Version:
*@Description:
**/
package optimizer

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"

	log "github.com/sirupsen/logrus"
	"github.com/tiktoken-go/tokenizer"
)

// =============================================================================
// Role priority for deterministic message ordering (Rule 1).
// Lower value = earlier position.
// =============================================================================
var rolePriority = map[string]int{
	"system":    0,
	"developer": 1,
	"assistant": 2,
	"tool":      3,
	"user":      4,
}

func roleString(v any) string {
	s, _ := v.(string)
	return s
}

func roleOrder(role string) int {
	if p, ok := rolePriority[strings.ToLower(strings.TrimSpace(role))]; ok {
		return p
	}
	return 100 // unknown roles go to the end
}

// =============================================================================
// Message field whitelist (Rule 6). Only these fields survive canonicalization.
// =============================================================================
var messageFieldWhitelist = map[string]bool{
	"role":       true,
	"content":    true,
	"tool_calls": true,
	"name":       true,
	// tool_call_id is kept on tool messages to match call→result.
	"tool_call_id": true,
}

// =============================================================================
// Regex for detecting markdown code blocks (Rule 10).
// =============================================================================
var codeBlockStart = regexp.MustCompile("^```")

// =============================================================================
// DeepSeek tokenizer — we cache a single instance for cache simulation.
// =============================================================================
var (
	tokenizerOnce sync.Once
	tokenizerEnc  tokenizer.Codec
	tokenizerErr  error
)

const deepSeekTokenizerModel = tokenizer.Cl100kBase

func getTokenizer() (tokenizer.Codec, error) {
	tokenizerOnce.Do(func() {
		tokenizerEnc, tokenizerErr = tokenizer.Get(deepSeekTokenizerModel)
	})
	return tokenizerEnc, tokenizerErr
}

// =============================================================================
// State for fingerprint tracking and cache simulation (Rules 11 & 12).
// Each conversation context (keyed by a structural hash of tools+system)
// gets its own state so concurrent conversations don't pollute each other's
// cache simulation metrics.
// =============================================================================
type promptState struct {
	mu              sync.Mutex
	lastFingerprint string
	lastTokens      []uint
	lastDriftReport string
}

// cacheSimStates stores per-conversation promptState keyed by a structural
// hash of the tools schema and system prompt.
var cacheSimStates sync.Map

// =============================================================================
// Rule 4+8: Recursive all-keys canonicalization via DFS.
// Every map[string]any is rebuilt with sorted keys; children processed first.
// =============================================================================

func canonicalizeAllKeys(v any) any {
	switch val := v.(type) {
	case map[string]any:
		for k, child := range val {
			val[k] = canonicalizeAllKeys(child)
		}
		return val
	case []any:
		for i, child := range val {
			val[i] = canonicalizeAllKeys(child)
		}
		return val
	default:
		return v
	}
}

// encodeSorted serializes a value to JSON with sorted keys and no extra whitespace.
func encodeSorted(v any) []byte {
	var buf bytes.Buffer
	writeSorted(&buf, v)
	return buf.Bytes()
}

func writeSorted(buf *bytes.Buffer, v any) {
	switch val := v.(type) {
	case map[string]any:
		buf.WriteByte('{')
		keys := make([]string, 0, len(val))
		for k := range val {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for i, k := range keys {
			if i > 0 {
				buf.WriteByte(',')
			}
			writeJSONString(buf, k)
			buf.WriteByte(':')
			writeSorted(buf, val[k])
		}
		buf.WriteByte('}')
	case []any:
		buf.WriteByte('[')
		for i, item := range val {
			if i > 0 {
				buf.WriteByte(',')
			}
			writeSorted(buf, item)
		}
		buf.WriteByte(']')
	case string:
		writeJSONString(buf, val)
	case json.Number:
		// Normalize number format: use the shortest representation.
		s := string(val)
		// json.Number preserves the original string; write as-is.
		buf.WriteString(s)
	case bool:
		if val {
			buf.WriteString("true")
		} else {
			buf.WriteString("false")
		}
	case nil:
		buf.WriteString("null")
	default:
		// Fallback: use encoding/json for unexpected types.
		fallback, _ := json.Marshal(val)
		buf.Write(fallback)
	}
}

func writeJSONString(buf *bytes.Buffer, s string) {
	// Use json.Marshal for correct JSON string escaping including Unicode.
	encoded, _ := json.Marshal(s)
	buf.Write(encoded)
}

// =============================================================================
// Rule 8: Sort tools by function name.
// =============================================================================

func sortToolsByName(tools any) any {
	arr, ok := tools.([]any)
	if !ok || len(arr) < 2 {
		return tools
	}
	type entry struct {
		idx  int
		name string
	}
	entries := make([]entry, 0, len(arr))
	for i, item := range arr {
		name := extractToolName(item)
		entries = append(entries, entry{idx: i, name: name})
	}
	if len(entries) != len(arr) {
		return tools
	}
	sort.SliceStable(entries, func(i, j int) bool {
		return entries[i].name < entries[j].name
	})
	sorted := make([]any, len(arr))
	for i, e := range entries {
		sorted[i] = arr[e.idx]
	}
	return sorted
}

func extractToolName(item any) string {
	m, ok := item.(map[string]any)
	if !ok {
		return ""
	}
	if fn, ok := m["function"]; ok {
		if fnMap, ok := fn.(map[string]any); ok {
			if n, ok := fnMap["name"]; ok {
				if s, ok := n.(string); ok {
					return s
				}
			}
		}
	}
	if t, ok := m["type"]; ok {
		if s, ok := t.(string); ok {
			return s
		}
	}
	return ""
}

// =============================================================================
// Rules 1-3, 6-7, 9: Message pipeline.
// =============================================================================

func canonicalizeMessagesDeep(msgs any) any {
	arr, ok := msgs.([]any)
	if !ok || len(arr) == 0 {
		return msgs
	}

	// --- Step 1: Whitelist fields per message (Rule 6). ---
	cleaned := make([]map[string]any, 0, len(arr))
	for _, raw := range arr {
		m, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		clean := make(map[string]any, len(m))
		for k, v := range m {
			if messageFieldWhitelist[k] {
				clean[k] = v
			}
		}
		if _, hasRole := clean["role"]; !hasRole {
			continue
		}
		cleaned = append(cleaned, clean)
	}

	if len(cleaned) == 0 {
		return arr
	}

	// --- Step 2: Canonicalize tool call arguments (Rule 5). ---
	for _, m := range cleaned {
		if tc, ok := m["tool_calls"]; ok {
			m["tool_calls"] = canonicalizeArguments(tc)
		}
	}

	// --- Step 3: ToolCall Rewrite (Rule 7). ---
	rewriteToolCallIDs(cleaned)

	// --- Step 4: Context Dedup (Rule 9). ---
	cleaned = dedupMessages(cleaned)

	// --- Step 5: System Prompt Merge (Rule 2). ---
	cleaned = mergeSystemMessages(cleaned)

	// --- Step 6: Assistant Merge (Rule 3). ---
	cleaned = mergeAssistantMessages(cleaned)

	// --- Step 7: Sort messages (Rule 1). ---
	cleaned = sortMessages(cleaned)

	// --- Step 8: Recursively canonicalize remaining nested structures. ---
	result := make([]any, len(cleaned))
	for i, m := range cleaned {
		result[i] = canonicalizeAllKeys(m)
	}
	return result
}

// =============================================================================
// Rule 6: Message field whitelist — already applied above.
// =============================================================================

// =============================================================================
// Rule 5: Canonicalize tool call arguments JSON.
// =============================================================================

func canonicalizeArguments(toolCalls any) any {
	arr, ok := toolCalls.([]any)
	if !ok {
		return toolCalls
	}
	for i, item := range arr {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		if fn, ok := m["function"]; ok {
			if fnMap, ok := fn.(map[string]any); ok {
				if args, ok := fnMap["arguments"]; ok {
					fnMap["arguments"] = canonicalizeArgString(args)
				}
			}
		}
		arr[i] = m
	}
	return arr
}

func canonicalizeArgString(args any) any {
	var argStr string
	switch v := args.(type) {
	case string:
		argStr = v
	case []byte:
		argStr = string(v)
	default:
		return args
	}
	argStr = strings.TrimSpace(argStr)
	if argStr == "" {
		return args
	}
	var parsed any
	if err := json.Unmarshal([]byte(argStr), &parsed); err != nil {
		// Not valid JSON; return original string.
		return argStr
	}
	// Re-encode with sorted keys.
	normalized := encodeSorted(canonicalizeAllKeys(parsed))
	return string(normalized)
}

// =============================================================================
// Rule 7: ToolCall Rewrite (call_1, call_2, ...).
// =============================================================================

func rewriteToolCallIDs(msgs []map[string]any) {
	// First pass: build the mapping from old ID → new ID, in message order.
	idMap := make(map[string]string)
	callSeq := 0

	for _, m := range msgs {
		tc, ok := m["tool_calls"]
		if !ok {
			continue
		}
		arr, ok := tc.([]any)
		if !ok {
			continue
		}
		for _, item := range arr {
			itemMap, ok := item.(map[string]any)
			if !ok {
				continue
			}
			oldID := ""
			if idVal, ok := itemMap["id"]; ok {
				if s, ok := idVal.(string); ok {
					oldID = strings.TrimSpace(s)
				}
			}
			if oldID == "" {
				continue
			}
			if _, exists := idMap[oldID]; !exists {
				callSeq++
				idMap[oldID] = fmt.Sprintf("call_%d", callSeq)
			}
		}
	}

	if len(idMap) == 0 {
		return
	}

	// Second pass: apply the mapping.
	for _, m := range msgs {
		// Rewrite tool_calls in assistant messages.
		if tc, ok := m["tool_calls"]; ok {
			if arr, ok := tc.([]any); ok {
				for _, item := range arr {
					if itemMap, ok := item.(map[string]any); ok {
						if idVal, ok := itemMap["id"]; ok {
							if s, ok := idVal.(string); ok {
								if newID, exists := idMap[strings.TrimSpace(s)]; exists {
									itemMap["id"] = newID
								}
							}
						}
					}
				}
			}
		}
		// Rewrite tool_call_id in tool messages.
		role, _ := m["role"].(string)
		if strings.ToLower(strings.TrimSpace(role)) == "tool" {
			if tcid, ok := m["tool_call_id"]; ok {
				if s, ok := tcid.(string); ok {
					if newID, exists := idMap[strings.TrimSpace(s)]; exists {
						m["tool_call_id"] = newID
					}
				}
			}
		}
	}
}

// =============================================================================
// Rule 9: Context Dedup via SHA256.
// =============================================================================

func dedupMessages(msgs []map[string]any) []map[string]any {
	seen := make(map[string]bool)
	result := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		h := hashMessageContent(m)
		if seen[h] {
			continue
		}
		seen[h] = true
		result = append(result, m)
	}
	return result
}

func hashMessageContent(m map[string]any) string {
	// Hash only the content, not tool_call_id or role.
	content, _ := m["content"]
	raw, _ := json.Marshal(content)
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

// =============================================================================
// Rule 2: System Prompt Merge. Merge consecutive system+developer into
// [system] [developer] pair at the front.
// =============================================================================

func mergeSystemMessages(msgs []map[string]any) []map[string]any {
	if len(msgs) == 0 {
		return msgs
	}

	// Separate system/developer messages from the rest.
	var systemContents []string
	var developerContents []string
	others := make([]map[string]any, 0, len(msgs))

	for _, m := range msgs {
		role, _ := m["role"].(string)
		role = strings.ToLower(strings.TrimSpace(role))
		switch role {
		case "system":
			if c := extractTextContent(m["content"]); c != "" {
				systemContents = append(systemContents, c)
			}
		case "developer":
			if c := extractTextContent(m["content"]); c != "" {
				developerContents = append(developerContents, c)
			}
		default:
			others = append(others, m)
		}
	}

	// Rebuild: system first (merged), then developer (merged), then others.
	result := make([]map[string]any, 0, len(msgs))
	if len(systemContents) > 0 {
		result = append(result, map[string]any{
			"role":    "system",
			"content": strings.Join(systemContents, "\n\n"),
		})
	}
	if len(developerContents) > 0 {
		result = append(result, map[string]any{
			"role":    "developer",
			"content": strings.Join(developerContents, "\n\n"),
		})
	}
	result = append(result, others...)
	return result
}

// =============================================================================
// Rule 3: Assistant Merge. Merge consecutive assistant messages.
// =============================================================================

func mergeAssistantMessages(msgs []map[string]any) []map[string]any {
	if len(msgs) < 2 {
		return msgs
	}
	result := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		if len(result) == 0 {
			result = append(result, m)
			continue
		}
		prev := result[len(result)-1]
		prevRole, _ := prev["role"].(string)
		curRole, _ := m["role"].(string)
		if strings.EqualFold(strings.TrimSpace(prevRole), "assistant") &&
			strings.EqualFold(strings.TrimSpace(curRole), "assistant") {
			// Merge content.
			prevContent := extractTextContent(prev["content"])
			curContent := extractTextContent(m["content"])
			if prevContent != "" && curContent != "" {
				prev["content"] = prevContent + "\n" + curContent
			} else if curContent != "" {
				prev["content"] = curContent
			}
			// Merge tool_calls: append after the last, dedup by function name.
			prevTC, _ := prev["tool_calls"].([]any)
			curTC, _ := m["tool_calls"].([]any)
			prev["tool_calls"] = mergeToolCalls(prevTC, curTC)
			continue
		}
		result = append(result, m)
	}
	return result
}

func mergeToolCalls(a, b []any) []any {
	if len(b) == 0 {
		return a
	}
	seen := make(map[string]bool)
	for _, item := range a {
		if m, ok := item.(map[string]any); ok {
			if fn, ok := m["function"]; ok {
				if fnMap, ok := fn.(map[string]any); ok {
					if name, ok := fnMap["name"]; ok {
						if s, ok := name.(string); ok {
							seen[s] = true
						}
					}
				}
			}
		}
	}
	for _, item := range b {
		if m, ok := item.(map[string]any); ok {
			if fn, ok := m["function"]; ok {
				if fnMap, ok := fn.(map[string]any); ok {
					if name, ok := fnMap["name"]; ok {
						if s, ok := name.(string); ok {
							if !seen[s] {
								a = append(a, item)
								seen[s] = true
							}
						}
					}
				}
			}
		}
	}
	return a
}

func extractTextContent(content any) string {
	switch v := content.(type) {
	case string:
		return v
	case []any:
		parts := make([]string, 0, len(v))
		for _, item := range v {
			if m, ok := item.(map[string]any); ok {
				if t, ok := m["type"].(string); ok && t == "text" {
					if text, ok := m["text"].(string); ok {
						parts = append(parts, text)
					}
				}
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

// =============================================================================
// Rule 1: Message ordering by role priority, then tool_call_id for tool msgs.
// =============================================================================

func sortMessages(msgs []map[string]any) []map[string]any {
	sorted := make([]map[string]any, len(msgs))
	copy(sorted, msgs)
	sort.SliceStable(sorted, func(i, j int) bool {
		ri := roleOrder(roleString(sorted[i]["role"]))
		rj := roleOrder(roleString(sorted[j]["role"]))
		if ri != rj {
			return ri < rj
		}
		// For tool messages, secondary sort by tool_call_id.
		if ri == rolePriority["tool"] {
			idi, _ := sorted[i]["tool_call_id"].(string)
			idj, _ := sorted[j]["tool_call_id"].(string)
			return idi < idj
		}
		return false
	})
	return sorted
}

// =============================================================================
// Rule 10: Whitespace normalization on string values inside JSON.
//   - Trim trailing whitespace per line
//   - Collapse consecutive blank lines
//   - CRLF → LF
//   - Preserve markdown code blocks (``` ... ```)
//   - Preserve indentation within code blocks
// =============================================================================

func normalizeWhitespaceInPlace(body []byte) []byte {
	// Re-parse the already-sorted JSON, normalize string values, re-encode.
	var root any
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	if err := dec.Decode(&root); err != nil {
		return body // not valid JSON, return as-is
	}

	normalized := normalizeStrings(root)
	return encodeSorted(normalized)
}

func normalizeStrings(v any) any {
	switch val := v.(type) {
	case map[string]any:
		for k, child := range val {
			val[k] = normalizeStrings(child)
		}
		return val
	case []any:
		for i, child := range val {
			val[i] = normalizeStrings(child)
		}
		return val
	case string:
		return normalizeStringContent(val)
	default:
		return v
	}
}

func normalizeStringContent(s string) string {
	if s == "" {
		return s
	}
	// CRLF → LF
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")

	lines := strings.Split(s, "\n")
	out := make([]string, 0, len(lines))
	inCodeBlock := false
	prevBlank := false

	for _, line := range lines {
		// Detect code block boundaries.
		if codeBlockStart.MatchString(strings.TrimSpace(line)) {
			inCodeBlock = !inCodeBlock
			out = append(out, line)
			prevBlank = false
			continue
		}

		if inCodeBlock {
			// Preserve code block content exactly (except CRLF already done).
			out = append(out, line)
			prevBlank = false
			continue
		}

		// Outside code blocks: trim trailing whitespace.
		line = strings.TrimRight(line, " \t")

		// Collapse consecutive blank lines.
		isBlank := strings.TrimSpace(line) == ""
		if isBlank {
			if prevBlank {
				continue
			}
			prevBlank = true
			out = append(out, "")
		} else {
			prevBlank = false
			out = append(out, line)
		}
	}

	// Remove trailing blank lines.
	for len(out) > 0 && out[len(out)-1] == "" {
		out = out[:len(out)-1]
	}

	return strings.Join(out, "\n")
}

// =============================================================================
// Rule 11: Prompt Fingerprint — log SHA256 and diff from last request.
// =============================================================================

// cacheSimKey computes a structural key for grouping related conversations.
// Uses tools + system prompt to group requests that share the same cache prefix.
func cacheSimKey(body []byte) string {
	var root map[string]any
	if err := json.Unmarshal(body, &root); err != nil {
		return "global" // fallback to shared state
	}

	h := sha256.New()

	// Hash tool names (deterministically sorted).
	if tools, ok := root["tools"]; ok {
		if arr, ok := tools.([]any); ok {
			names := make([]string, 0, len(arr))
			for _, t := range arr {
				if m, ok := t.(map[string]any); ok {
					if fn, ok := m["function"]; ok {
						if fnMap, ok := fn.(map[string]any); ok {
							if n, ok := fnMap["name"]; ok {
								if s, ok := n.(string); ok {
									names = append(names, s)
								}
							}
						}
					}
				}
			}
			sort.Strings(names)
			for _, n := range names {
				h.Write([]byte(n))
				h.Write([]byte{0})
			}
		}
	}

	// Hash system prompt content.
	if msgs, ok := root["messages"]; ok {
		if arr, ok := msgs.([]any); ok {
			for _, msg := range arr {
				if m, ok := msg.(map[string]any); ok {
					role, _ := m["role"].(string)
					role = strings.ToLower(strings.TrimSpace(role))
					if role == "system" || role == "developer" {
						if c, ok := m["content"]; ok {
							if s, ok := c.(string); ok {
								h.Write([]byte(s))
							}
						}
					}
				}
			}
		}
	}

	return hex.EncodeToString(h.Sum(nil)[:8])
}

func sha256sum(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:8]) // first 8 bytes → 16 hex chars
}

func logDriftIfChanged(fingerprint string, body []byte) {
	key := cacheSimKey(body)
	raw, _ := cacheSimStates.LoadOrStore(key, &promptState{})
	state := raw.(*promptState)

	state.mu.Lock()
	defer state.mu.Unlock()

	if state.lastFingerprint == "" {
		state.lastFingerprint = fingerprint
		log.WithField("fingerprint", fingerprint).Debug("deepseek cache optimizer: initial prompt fingerprint")
		return
	}

	if fingerprint == state.lastFingerprint {
		return // identical, no drift
	}

	// Compute drift report.
	report := computeDriftReport(body)
	if report != "" {
		log.WithFields(log.Fields{
			"fingerprint":      fingerprint,
			"prev_fingerprint": state.lastFingerprint,
			"drift":            report,
		}).Info("deepseek cache optimizer: prompt drift detected")
	}

	state.lastFingerprint = fingerprint
	state.lastDriftReport = report
}

// computeDriftReport extracts top-level changed sections by re-parsing and
// comparing structural keys against the previous fingerprint context.
// Returns a compact human-readable string like "Tool Schema:browser,filesystem Arguments:timeout,url".
func computeDriftReport(body []byte) string {
	// Extract structural summary: which tools, which argument keys, which message roles.
	var root map[string]any
	if err := json.Unmarshal(body, &root); err != nil {
		return ""
	}

	parts := make([]string, 0)

	if tools, ok := root["tools"]; ok {
		if arr, ok := tools.([]any); ok {
			names := make([]string, 0, len(arr))
			for _, t := range arr {
				if m, ok := t.(map[string]any); ok {
					if fn, ok := m["function"]; ok {
						if fnMap, ok := fn.(map[string]any); ok {
							if n, ok := fnMap["name"]; ok {
								if s, ok := n.(string); ok {
									names = append(names, s)
								}
							}
						}
					}
				}
			}
			if len(names) > 0 {
				parts = append(parts, "Tool Schema:"+strings.Join(names, ","))
			}
		}
	}

	// Collect argument keys across all tool call arguments.
	argKeys := make(map[string]bool)
	if msgs, ok := root["messages"]; ok {
		if arr, ok := msgs.([]any); ok {
			for _, msg := range arr {
				if m, ok := msg.(map[string]any); ok {
					if tc, ok := m["tool_calls"]; ok {
						if tcArr, ok := tc.([]any); ok {
							for _, tcItem := range tcArr {
								if tcMap, ok := tcItem.(map[string]any); ok {
									if fn, ok := tcMap["function"]; ok {
										if fnMap, ok := fn.(map[string]any); ok {
											if args, ok := fnMap["arguments"]; ok {
												collectArgKeys(args, argKeys)
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	if len(argKeys) > 0 {
		keysList := sortedKeys(argKeys)
		parts = append(parts, "Arguments:"+strings.Join(keysList, ","))
	}

	// Message role summary.
	roleCounts := make(map[string]int)
	if msgs, ok := root["messages"]; ok {
		if arr, ok := msgs.([]any); ok {
			for _, msg := range arr {
				if m, ok := msg.(map[string]any); ok {
					if role, ok := m["role"].(string); ok {
						roleCounts[strings.ToLower(strings.TrimSpace(role))]++
					}
				}
			}
		}
	}
	if len(roleCounts) > 0 {
		rl := sortedKeys(roleCounts)
		parts = append(parts, "Messages:"+strings.Join(rl, ","))
	}

	return strings.Join(parts, " ")
}

func collectArgKeys(args any, keys map[string]bool) {
	var argStr string
	switch v := args.(type) {
	case string:
		argStr = v
	default:
		return
	}
	var parsed map[string]any
	if json.Unmarshal([]byte(argStr), &parsed) == nil {
		for k := range parsed {
			keys[k] = true
		}
	}
}

func sortedKeys(m any) []string {
	switch val := m.(type) {
	case map[string]bool:
		keys := make([]string, 0, len(val))
		for k := range val {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		return keys
	case map[string]int:
		keys := make([]string, 0, len(val))
		for k := range val {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		return keys
	}
	return nil
}

// =============================================================================
// Rule 12: Cache Simulation — tokenize, compare with last, estimate hit rate.
// =============================================================================

func simulateCache(body []byte) {
	enc, err := getTokenizer()
	if err != nil {
		return
	}

	// Tokenize the serialized JSON body as a string.
	text := string(body)
	ids, _, errTok := enc.Encode(text)
	if errTok != nil {
		return
	}
	tokens := make([]uint, len(ids))
	for i, id := range ids {
		tokens[i] = uint(id)
	}

	key := cacheSimKey(body)
	raw, _ := cacheSimStates.LoadOrStore(key, &promptState{})
	state := raw.(*promptState)

	state.mu.Lock()
	defer state.mu.Unlock()

	if len(state.lastTokens) == 0 {
		state.lastTokens = tokens
		return
	}

	// Compute common prefix length.
	common := 0
	maxCommon := min(len(tokens), len(state.lastTokens))
	for common < maxCommon && state.lastTokens[common] == tokens[common] {
		common++
	}

	totalTokens := len(tokens)
	var estimatedCache float64
	if totalTokens > 0 {
		estimatedCache = float64(common) / float64(totalTokens) * 100
	}

	log.WithFields(log.Fields{
		"total_tokens":    totalTokens,
		"cached_tokens":   common,
		"estimated_cache": fmt.Sprintf("%.0f%%", estimatedCache),
	}).Debug("deepseek cache optimizer: cache simulation")

	if estimatedCache < 70 {
		log.WithFields(log.Fields{
			"estimated_cache": fmt.Sprintf("%.0f%%", estimatedCache),
			"total_tokens":    totalTokens,
			"cached_tokens":   common,
			"drift":           state.lastDriftReport,
		}).Warn("deepseek cache optimizer: Large Prompt Drift — low cache hit rate expected")
	}

	state.lastTokens = tokens
}
