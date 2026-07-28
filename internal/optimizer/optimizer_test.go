package optimizer

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestRoleOrder(t *testing.T) {
	tests := []struct {
		role string
		want int
	}{
		{"system", 0},
		{"developer", 1},
		{"assistant", 2},
		{"tool", 3},
		{"user", 4},
		{"SYSTEM", 0},
		{"  system  ", 0},
		{"unknown", 100},
		{"", 100},
	}
	for _, tt := range tests {
		got := roleOrder(tt.role)
		if got != tt.want {
			t.Errorf("roleOrder(%q) = %d, want %d", tt.role, got, tt.want)
		}
	}
}

func TestEncodeSorted_MapKeysSorted(t *testing.T) {
	input := map[string]any{
		"z": 1,
		"a": 2,
		"m": 3,
	}
	output := encodeSorted(input)
	outStr := string(output)
	aIdx := strings.Index(outStr, "\"a\"")
	mIdx := strings.Index(outStr, "\"m\"")
	zIdx := strings.Index(outStr, "\"z\"")
	if aIdx < 0 || mIdx < 0 || zIdx < 0 {
		t.Fatalf("missing key in output: %s", outStr)
	}
	if !(aIdx < mIdx && mIdx < zIdx) {
		t.Errorf("keys not sorted in JSON: %s", outStr)
	}
}

func TestEncodeSorted_NestedMaps(t *testing.T) {
	input := map[string]any{
		"outer": map[string]any{
			"z": 1,
			"a": 2,
		},
	}
	output := encodeSorted(input)
	if !strings.Contains(string(output), `"a":2`) {
		t.Errorf("nested keys not sorted: %s", string(output))
	}
	if !strings.Contains(string(output), `"z":1`) {
		t.Errorf("nested key z missing: %s", string(output))
	}
	// a should come before z
	aIdx := strings.Index(string(output), `"a"`)
	zIdx := strings.Index(string(output), `"z"`)
	if aIdx > zIdx {
		t.Errorf("nested keys not sorted: a at %d, z at %d", aIdx, zIdx)
	}
}

func TestEncodeSorted_ArrayPreservation(t *testing.T) {
	input := map[string]any{
		"items": []any{3, 1, 2},
	}
	output := encodeSorted(input)
	if !strings.Contains(string(output), `[3,1,2]`) {
		t.Errorf("array order not preserved: %s", string(output))
	}
}

func TestEncodeSorted_NullBoolNumber(t *testing.T) {
	input := map[string]any{
		"nullval":  nil,
		"boolval":  true,
		"numval":   json.Number("42"),
	}
	output := encodeSorted(input)
	if !strings.Contains(string(output), `null`) {
		t.Errorf("null value not encoded: %s", string(output))
	}
	if !strings.Contains(string(output), `true`) {
		t.Errorf("bool value not encoded: %s", string(output))
	}
	if !strings.Contains(string(output), `42`) {
		t.Errorf("number value not encoded: %s", string(output))
	}
}

func TestSortToolsByName_AlreadySorted(t *testing.T) {
	tools := []any{
		map[string]any{"function": map[string]any{"name": "a_tool"}},
		map[string]any{"function": map[string]any{"name": "b_tool"}},
	}
	result := sortToolsByName(tools)
	arr := result.([]any)
	if len(arr) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(arr))
	}
}

func TestSortToolsByName_ReverseSorted(t *testing.T) {
	tools := []any{
		map[string]any{"function": map[string]any{"name": "z_tool"}},
		map[string]any{"function": map[string]any{"name": "a_tool"}},
	}
	result := sortToolsByName(tools)
	arr := result.([]any)
	m0 := arr[0].(map[string]any)
	m1 := arr[1].(map[string]any)
	if m0["function"].(map[string]any)["name"] != "a_tool" {
		t.Errorf("first tool should be a_tool, got %v", m0["function"])
	}
	if m1["function"].(map[string]any)["name"] != "z_tool" {
		t.Errorf("second tool should be z_tool, got %v", m1["function"])
	}
}

func TestSortToolsByName_SingleElement(t *testing.T) {
	tools := []any{
		map[string]any{"function": map[string]any{"name": "only_tool"}},
	}
	result := sortToolsByName(tools)
	arr := result.([]any)
	if len(arr) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(arr))
	}
}

func TestExtractToolName(t *testing.T) {
	tests := []struct {
		name string
		item any
		want string
	}{
		{"function with name", map[string]any{"function": map[string]any{"name": "my_tool"}}, "my_tool"},
		{"type field", map[string]any{"type": "code_interpreter"}, "code_interpreter"},
		{"empty map", map[string]any{}, ""},
		{"nil", nil, ""},
	}
	for _, tt := range tests {
		got := extractToolName(tt.item)
		if got != tt.want {
			t.Errorf("extractToolName(%s) = %q, want %q", tt.name, got, tt.want)
		}
	}
}

func TestCanonicalizeArguments(t *testing.T) {
	input := []any{map[string]any{
		"id": "c1",
		"function": map[string]any{
			"name":      "test",
			"arguments": `{"z":3,"a":1,"m":2}`,
		},
	}}
	result := canonicalizeArguments(input)
	arr := result.([]any)
	m := arr[0].(map[string]any)
	args := m["function"].(map[string]any)["arguments"].(string)
	if !strings.Contains(args, `"a":1`) {
		t.Errorf("argument 'a' should be first, got: %s", args)
	}
}

func TestCanonicalizeArguments_InvalidJSON(t *testing.T) {
	input := []any{map[string]any{
		"id": "c1",
		"function": map[string]any{
			"name":      "test",
			"arguments": `not-json`,
		},
	}}
	result := canonicalizeArguments(input)
	arr := result.([]any)
	m := arr[0].(map[string]any)
	args := m["function"].(map[string]any)["arguments"].(string)
	if args != "not-json" {
		t.Errorf("invalid JSON arguments should be preserved as-is, got: %s", args)
	}
}

func TestRewriteToolCallIDs(t *testing.T) {
	msgs := []map[string]any{
		{"role": "assistant", "tool_calls": []any{
			map[string]any{"id": "call_old_1", "function": map[string]any{"name": "read"}},
			map[string]any{"id": "call_old_2", "function": map[string]any{"name": "write"}},
		}},
		{"role": "tool", "tool_call_id": "call_old_1", "content": "result1"},
		{"role": "tool", "tool_call_id": "call_old_2", "content": "result2"},
	}
	rewriteToolCallIDs(msgs)

	// Check tool calls were rewritten to call_N
	for _, m := range msgs {
		if tc, ok := m["tool_calls"]; ok {
			for _, item := range tc.([]any) {
				id := item.(map[string]any)["id"].(string)
				if !strings.HasPrefix(id, "call_") {
					t.Errorf("tool call id = %q, want call_N prefix", id)
				}
				if strings.Contains(id, "old") {
					t.Errorf("tool call id still contains old prefix: %q", id)
				}
			}
		}
		if m["role"] == "tool" {
			tcid := m["tool_call_id"].(string)
			if !strings.HasPrefix(tcid, "call_") {
				t.Errorf("tool_call_id = %q, want call_N prefix", tcid)
			}
		}
	}
}

func TestDedupMessages(t *testing.T) {
	msgs := []map[string]any{
		{"role": "user", "content": "Hello"},
		{"role": "user", "content": "World"},
		{"role": "user", "content": "Hello"}, // duplicate
	}
	result := dedupMessages(msgs)
	if len(result) != 2 {
		t.Errorf("expected 2 messages after dedup, got %d", len(result))
	}
}

func TestMergeSystemMessages(t *testing.T) {
	msgs := []map[string]any{
		{"role": "system", "content": "Rule 1"},
		{"role": "system", "content": "Rule 2"},
		{"role": "user", "content": "Hello"},
	}
	result := mergeSystemMessages(msgs)
	if len(result) != 2 {
		t.Fatalf("expected 2 messages after merge, got %d", len(result))
	}
	if result[0]["role"] != "system" {
		t.Errorf("first message role = %q, want system", result[0]["role"])
	}
	content := result[0]["content"].(string)
	if !strings.Contains(content, "Rule 1") || !strings.Contains(content, "Rule 2") {
		t.Errorf("merged content = %q, want both rules", content)
	}
}

func TestMergeAssistantMessages(t *testing.T) {
	msgs := []map[string]any{
		{"role": "assistant", "content": "First part"},
		{"role": "assistant", "content": "Second part"},
		{"role": "user", "content": "Hello"},
	}
	result := mergeAssistantMessages(msgs)
	if len(result) != 2 {
		t.Fatalf("expected 2 messages after merge, got %d", len(result))
	}
	content := result[0]["content"].(string)
	if !strings.Contains(content, "First part") || !strings.Contains(content, "Second part") {
		t.Errorf("merged content = %q, want both parts", content)
	}
}

func TestNormalizeWhitespace(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "crlf to lf",
			input:    "line1\r\nline2",
			expected: "line1\nline2",
		},
		{
			name:     "trailing spaces",
			input:    "line1   \nline2  ",
			expected: "line1\nline2",
		},
		{
			name:     "collapse blank lines",
			input:    "a\n\n\nb",
			expected: "a\n\nb",
		},
		{
			name:     "preserve code blocks",
			input:    "text\n```\ncode with  spaces  \n\n\nmore code\n```\nend",
			expected: "text\n```\ncode with  spaces  \n\n\nmore code\n```\nend",
		},
		{
			name:     "trailing blank lines removed",
			input:    "a\n\n",
			expected: "a",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := normalizeStringContent(tt.input)
			if result != tt.expected {
				t.Errorf("normalizeStringContent(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestNormalizeWhitespaceInPlace(t *testing.T) {
	input := []byte(`{"messages":[{"role":"user","content":"Hi   \n\n\nThere"}]}`)
	output := normalizeWhitespaceInPlace(input)
	var root map[string]any
	json.Unmarshal(output, &root)
	content := root["messages"].([]any)[0].(map[string]any)["content"].(string)
	if !strings.Contains(content, "Hi") || !strings.Contains(content, "There") {
		t.Errorf("content corrupted: %q", content)
	}
}

func TestExtractTextContent(t *testing.T) {
	t.Run("string content", func(t *testing.T) {
		result := extractTextContent("hello world")
		if result != "hello world" {
			t.Errorf("got %q, want 'hello world'", result)
		}
	})
	t.Run("array content", func(t *testing.T) {
		result := extractTextContent([]any{
			map[string]any{"type": "text", "text": "hello"},
			map[string]any{"type": "image_url", "image_url": "http://example.com/img.png"},
			map[string]any{"type": "text", "text": "world"},
		})
		if result != "hello\nworld" {
			t.Errorf("got %q, want 'hello\\nworld'", result)
		}
	})
	t.Run("nil content", func(t *testing.T) {
		result := extractTextContent(nil)
		if result != "" {
			t.Errorf("got %q, want ''", result)
		}
	})
}

func TestSortedKeys(t *testing.T) {
	result := sortedKeys(map[string]int{"z": 1, "a": 2, "m": 3})
	expected := []string{"a", "m", "z"}
	for i, k := range result {
		if k != expected[i] {
			t.Errorf("key[%d] = %q, want %q", i, k, expected[i])
		}
	}
}

func TestCacheSimKey_DifferentToolsDifferentKey(t *testing.T) {
	body1 := []byte(`{"tools":[{"function":{"name":"z_tool"}}],"messages":[{"role":"user","content":"hi"}]}`)
	body2 := []byte(`{"tools":[{"function":{"name":"a_tool"}}],"messages":[{"role":"user","content":"hi"}]}`)

	key1 := cacheSimKey(body1)
	key2 := cacheSimKey(body2)

	if key1 == key2 {
		t.Errorf("different tools should produce different keys: %s vs %s", key1, key2)
	}
}

func TestCacheSimKey_SameToolsSameKey(t *testing.T) {
	body1 := []byte(`{"tools":[{"function":{"name":"read"}},{"function":{"name":"write"}}],"messages":[{"role":"user","content":"Hello"}]}`)
	body2 := []byte(`{"tools":[{"function":{"name":"write"}},{"function":{"name":"read"}}],"messages":[{"role":"user","content":"World"}]}`)

	key1 := cacheSimKey(body1)
	key2 := cacheSimKey(body2)

	if key1 == "" {
		t.Errorf("key should not be empty")
	}
	if key1 != key2 {
		t.Errorf("same tools, different user messages should produce same key: %s vs %s", key1, key2)
	}
}

func TestCacheSimKey_DifferentSystemDifferentKey(t *testing.T) {
	body1 := []byte(`{"messages":[{"role":"system","content":"Be concise"},{"role":"user","content":"hi"}]}`)
	body2 := []byte(`{"messages":[{"role":"system","content":"Be verbose"},{"role":"user","content":"hi"}]}`)

	key1 := cacheSimKey(body1)
	key2 := cacheSimKey(body2)

	if key1 == key2 {
		t.Errorf("different system prompts should produce different keys: %s vs %s", key1, key2)
	}
}

func TestCacheSimKey_InvalidJSON(t *testing.T) {
	key := cacheSimKey([]byte(`{invalid}`))
	if key != "global" {
		t.Errorf("invalid JSON should return 'global', got %q", key)
	}
}
