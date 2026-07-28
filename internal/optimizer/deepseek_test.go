package optimizer

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestDeepSeekApplyToRequest_SortsKeys(t *testing.T) {
	input := []byte(`{"b":2,"a":1,"messages":[{"role":"user","content":"hi","extra":"drop"}],"c":3}`)
	output := DeepSeekApplyToRequest("deepseek", input)
	if !json.Valid(output) { t.Fatalf("output is not valid JSON: %s", string(output)) }
	outStr := string(output)
	aIdx := strings.Index(outStr, `"a"`)
	bIdx := strings.Index(outStr, `"b"`)
	cIdx := strings.Index(outStr, `"c"`)
	mIdx := strings.Index(outStr, `"messages"`)
	if aIdx < 0 || bIdx < 0 || cIdx < 0 || mIdx < 0 {
		t.Fatalf("missing key in output: %s", outStr)
	}
	if !(aIdx < bIdx && bIdx < cIdx && cIdx < mIdx) {
		t.Errorf("keys not sorted in JSON output: %s", outStr)
	}
}

func TestDeepSeekApplyToRequest_PreservesSemanticFields(t *testing.T) {
	input := []byte(`{"model":"deepseek-chat","messages":[{"role":"system","content":"Be helpful"},{"role":"user","content":"Hello"}],"temperature":0.7,"max_tokens":1000}`)
	output := DeepSeekApplyToRequest("deepseek", input)
	if !json.Valid(output) { t.Fatalf("invalid JSON: %s", string(output)) }
	var root map[string]any
	json.Unmarshal(output, &root)
	if root["messages"] == nil { t.Error("messages removed") }
	if root["temperature"] == nil { t.Error("temperature removed") }
	if root["max_tokens"] == nil { t.Error("max_tokens removed") }
}

func TestDeepSeekApplyToRequest_StripsStreamOptions(t *testing.T) {
	input := []byte(`{"model":"deepseek-chat","stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}`)
	output := DeepSeekApplyToRequest("deepseek", input)
	if strings.Contains(string(output), "stream_options") {
		t.Errorf("stream_options should be stripped, got: %s", string(output))
	}
}

func TestDeepSeekApplyToRequest_StripsUserField(t *testing.T) {
	input := []byte(`{"model":"deepseek-chat","user":"some-end-user-id","messages":[{"role":"user","content":"hi"}]}`)
	output := DeepSeekApplyToRequest("deepseek", input)
	var root map[string]any
	json.Unmarshal(output, &root)
	if _, exists := root["user"]; exists {
		t.Errorf("user field should be stripped, got: %s", string(output))
	}
}

func TestDeepSeekApplyToRequest_SortsMessagesByRole(t *testing.T) {
	input := []byte(`{"messages":[{"role":"user","content":"Hello"},{"role":"system","content":"Be helpful"},{"role":"assistant","content":"Hi there"}]}`)
	output := DeepSeekApplyToRequest("deepseek", input)
	var root map[string]any
	json.Unmarshal(output, &root)
	msgs := root["messages"].([]any)
	roles := make([]string, len(msgs))
	for i, m := range msgs {
		mm := m.(map[string]any)
		roles[i] = mm["role"].(string)
	}
	if roles[0] != "system" {
		t.Errorf("first role = %q, want system; roles=%v", roles[0], roles)
	}
	if roles[len(roles)-1] != "user" {
		t.Errorf("last role = %q, want user; roles=%v", roles[len(roles)-1], roles)
	}
}

func TestDeepSeekApplyToRequest_SortsToolsByName(t *testing.T) {
	input := []byte(`{"messages":[{"role":"user","content":"hi"}],"tools":[{"function":{"name":"z_tool"},"type":"function"},{"function":{"name":"a_tool"},"type":"function"},{"function":{"name":"m_tool"},"type":"function"}]}`)
	output := DeepSeekApplyToRequest("deepseek", input)
	var root map[string]any
	json.Unmarshal(output, &root)
	tools := root["tools"].([]any)
	names := make([]string, len(tools))
	for i, t := range tools {
		tt := t.(map[string]any)
		names[i] = tt["function"].(map[string]any)["name"].(string)
	}
	expected := []string{"a_tool", "m_tool", "z_tool"}
	for i := range expected {
		if names[i] != expected[i] {
			t.Errorf("tool[%d] = %q, want %q; got %v", i, names[i], expected[i], names)
		}
	}
}

func TestDeepSeekApplyToRequest_RewritesToolCallIDs(t *testing.T) {
	input := []byte(`{"messages":[{"role":"assistant","tool_calls":[{"id":"xyz789","function":{"name":"read","arguments":"{}"}}],"content":null},{"role":"tool","tool_call_id":"xyz789","content":"result"}]}`)
	output := DeepSeekApplyToRequest("deepseek", input)
	var root map[string]any
	json.Unmarshal(output, &root)
	msgs := root["messages"].([]any)
	for _, m := range msgs {
		mm := m.(map[string]any)
		if mm["role"] == "assistant" {
			tcs := mm["tool_calls"].([]any)
			tc := tcs[0].(map[string]any)
			if !strings.HasPrefix(tc["id"].(string), "call_") {
				t.Errorf("tool call ID = %q, want call_N prefix", tc["id"])
			}
		}
		if mm["role"] == "tool" {
			if !strings.HasPrefix(mm["tool_call_id"].(string), "call_") {
				t.Errorf("tool_call_id = %q, want call_N prefix", mm["tool_call_id"])
			}
		}
	}
}

func TestDeepSeekApplyToRequest_NormalizesWhitespace(t *testing.T) {
	// Build input via json.Marshal to avoid escaping confusion.
	rawContent := "Line1   \n  \n\nLine2  \n  Line3   "
	input, _ := json.Marshal(map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": rawContent},
		},
	})
	output := DeepSeekApplyToRequest("deepseek", input)
	var root map[string]any
	json.Unmarshal(output, &root)
	msgs := root["messages"].([]any)
	content := msgs[0].(map[string]any)["content"].(string)
	// Trailing whitespace should be trimmed by normalization.
	if strings.Contains(content, "   ") {
		t.Errorf("trailing whitespace still present after normalization: %q", content)
	}
	// Consecutive blank lines should be collapsed (3 -> 1 blank line).
	if strings.Contains(content, "\n\n\n") {
		t.Errorf("consecutive blank lines not collapsed: %q", content)
	}
	// Core content should be preserved.
	if !strings.Contains(content, "Line1") || !strings.Contains(content, "Line2") || !strings.Contains(content, "Line3") {
		t.Errorf("core content lost: %q", content)
	}
}

func TestDeepSeekApplyToRequest_DedupMessages(t *testing.T) {
	input := []byte(`{"messages":[{"role":"user","content":"Hello"},{"role":"user","content":"Hello"}]}`)
	output := DeepSeekApplyToRequest("deepseek", input)
	var root map[string]any
	json.Unmarshal(output, &root)
	msgs := root["messages"].([]any)
	if len(msgs) != 1 {
		t.Errorf("expected 1 message after dedup, got %d: %v", len(msgs), msgs)
	}
}

func TestDeepSeekApplyToRequest_CanonicalizeArguments(t *testing.T) {
	// Build via json.Marshal so nested escaping is correct.
	rawArgs := "{\"b\":2,\"a\":1}"
	input, _ := json.Marshal(map[string]any{
		"messages": []any{
			map[string]any{
				"role": "assistant",
				"tool_calls": []any{
					map[string]any{
						"id": "c1",
						"function": map[string]any{
							"name":      "read",
							"arguments": rawArgs,
						},
					},
				},
			},
			map[string]any{
				"role":         "tool",
				"tool_call_id": "c1",
				"content":      "result",
			},
		},
	})
	output := DeepSeekApplyToRequest("deepseek", input)
	var root map[string]any
	json.Unmarshal(output, &root)
	msgs := root["messages"].([]any)
	for _, m := range msgs {
		mm := m.(map[string]any)
		if mm["role"] == "assistant" {
			tcs := mm["tool_calls"].([]any)
			if len(tcs) == 0 {
				t.Fatal("assistant message has no tool_calls")
			}
			tc := tcs[0].(map[string]any)
			fn, ok := tc["function"].(map[string]any)
			if !ok {
				t.Fatalf("function not found in tool_call: %v", tc)
			}
			args := fn["arguments"].(string)
			if !strings.Contains(args, `"a":1`) || !strings.Contains(args, `"b":2`) {
				t.Errorf("arguments not canonicalized: %s", args)
			}
			if strings.Index(args, `"a"`) > strings.Index(args, `"b"`) {
				t.Errorf("argument keys not sorted: %s", args)
			}
		}
	}
}

func TestDeepSeekApplyToRequest_NotDeepSeek(t *testing.T) {
	input := []byte(`{"b":2,"a":1}`)
	output := DeepSeekApplyToRequest("claude", input)
	if string(output) != string(input) {
		t.Errorf("non-DeepSeek provider should return body unchanged, got %s", string(output))
	}
}

func TestDeepSeekApplyToRequest_OpenAICompatibleDeepSeek(t *testing.T) {
	input := []byte(`{"b":2,"a":1,"messages":[{"role":"user","content":"hi"}]}`)
	output := DeepSeekApplyToRequest("openai-compatible-deepseek-custom", input)
	outStr := string(output)
	aIdx := strings.Index(outStr, `"a"`)
	bIdx := strings.Index(outStr, `"b"`)
	mIdx := strings.Index(outStr, `"messages"`)
	if aIdx < 0 || bIdx < 0 || mIdx < 0 {
		t.Fatalf("missing key in output: %s", outStr)
	}
	if !(aIdx < bIdx && bIdx < mIdx) {
		t.Errorf("keys not sorted: %s", outStr)
	}
}

func TestDeepSeekApplyToRequest_MergesSystemMessages(t *testing.T) {
	input := []byte(`{"messages":[{"role":"system","content":"Rule 1"},{"role":"system","content":"Rule 2"},{"role":"user","content":"Hello"}]}`)
	output := DeepSeekApplyToRequest("deepseek", input)
	var root map[string]any
	json.Unmarshal(output, &root)
	msgs := root["messages"].([]any)
	if len(msgs) != 2 {
		t.Errorf("expected 2 messages after merge, got %d: %v", len(msgs), msgs)
	}
	first := msgs[0].(map[string]any)
	if first["role"] != "system" {
		t.Errorf("first message role = %q, want system", first["role"])
	}
	content := first["content"].(string)
	if !strings.Contains(content, "Rule 1") || !strings.Contains(content, "Rule 2") {
		t.Errorf("merged content missing rules: %q", content)
	}
}

func TestDeepSeekApplyToRequest_HandlesEmptyBody(t *testing.T) {
	output := DeepSeekApplyToRequest("deepseek", []byte{})
	if string(output) != "" {
		t.Errorf("empty body should return empty, got: %s", string(output))
	}
}

func TestDeepSeekApplyToRequest_HandlesInvalidJSON(t *testing.T) {
	input := []byte("{invalid json}")
	output := DeepSeekApplyToRequest("deepseek", input)
	if string(output) != string(input) {
		t.Errorf("invalid JSON should return original body, got: %s", string(output))
	}
}

func TestCanonicalizeDeepSeekRequest_Deterministic(t *testing.T) {
	input1 := []byte(`{"messages":[{"role":"user","content":"hi"}],"model":"deepseek-chat","temperature":0.7,"max_tokens":100}`)
	input2 := []byte(`{"max_tokens":100,"temperature":0.7,"model":"deepseek-chat","messages":[{"role":"user","content":"hi"}]}`)
	out1, _ := canonicalizeDeepSeekRequest(input1)
	out2, _ := canonicalizeDeepSeekRequest(input2)
	if string(out1) != string(out2) {
		t.Errorf("deterministic canonicalization failed:\\n  out1=%s\\n  out2=%s", string(out1), string(out2))
	}
}

func TestCanonicalizeDeepSeekRequest_PreservesCodeBlocks(t *testing.T) {
	// Build JSON with code block entirely via json.Marshal to avoid backtick issues.
	// The backtick-containing Go raw strings here are only for string JOIN elements,
	// not for the overall JSON raw literal, so they work correctly.
	codeBlock := "`" + "`" + "`go" + "\n" +
		"func main() {\n" +
		"    fmt.Println(\"hello\")\n" +
		"}\n" +
		"`" + "`" + "`" + "\n"
	content := "Here's code:\n" + codeBlock + "\nWhat do you think?"
	input, _ := json.Marshal(map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": content},
		},
	})
	output, err := canonicalizeDeepSeekRequest(input)
	if err != nil {
		t.Fatalf("canonicalization failed: %v", err)
	}
	var root map[string]any
	json.Unmarshal(output, &root)
	msgs := root["messages"].([]any)
	got := msgs[0].(map[string]any)["content"].(string)
	if !strings.Contains(got, "    fmt.Println") {
		t.Errorf("code block indentation not preserved: %q", got)
	}
}
func TestIsDeepSeekProvider(t *testing.T) {
	tests := []struct { provider string; want bool }{
		{"deepseek", true},
		{"DeepSeek", true},
		{"  deepseek  ", true},
		{"openai-compatible-deepseek", true},
		{"openai-compatible-deepseek-custom", true},
		{"openai-compatible", false},
		{"claude", false},
		{"", false},
	}
	for _, tt := range tests {
		got := isDeepSeekProvider(tt.provider)
		if got != tt.want {
			t.Errorf("isDeepSeekProvider(%q) = %v, want %v", tt.provider, got, tt.want)
		}
	}
}
