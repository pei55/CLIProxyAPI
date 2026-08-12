package executor

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

const openAICompatReasoningHistoryPayload = `{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"answer","reasoning_content":"prior reasoning","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_1","content":"ok"}]}`

func newOpenAICompatReasoningTestExecutor(t *testing.T, upstreamHandler http.HandlerFunc) (*OpenAICompatExecutor, *httptest.Server) {
	t.Helper()
	server := httptest.NewServer(upstreamHandler)
	t.Cleanup(server.Close)
	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	return executor, server
}

func TestOpenAICompatExecutorPreservesReasoningContentNonStream(t *testing.T) {
	var upstreamBody []byte
	executor, server := newOpenAICompatReasoningTestExecutor(t, func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok","reasoning_content":"prior reasoning"},"finish_reason":"stop"}]}`))
	})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": server.URL,
			"api_key":  "test-key",
		},
	}
	request := cliproxyexecutor.Request{
		Model:   "deepseek-v4-flash",
		Payload: []byte(openAICompatReasoningHistoryPayload),
	}
	options := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatOpenAI,
		ResponseFormat: sdktranslator.FormatOpenAI,
	}

	if _, errExecute := executor.Execute(context.Background(), auth, request, options); errExecute != nil {
		t.Fatalf("Execute error: %v", errExecute)
	}
	if got := gjson.GetBytes(upstreamBody, "messages.1.reasoning_content").String(); got != "prior reasoning" {
		t.Fatalf("upstream reasoning_content = %q, want %q; body=%s", got, "prior reasoning", upstreamBody)
	}
}

func TestOpenAICompatExecutorPreservesReasoningContentStream(t *testing.T) {
	var upstreamBody []byte
	executor, server := newOpenAICompatReasoningTestExecutor(t, func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"reasoning_content\":\"prior reasoning\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n")
	})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": server.URL,
			"api_key":  "test-key",
		},
	}
	request := cliproxyexecutor.Request{
		Model:   "deepseek-v4-flash",
		Payload: []byte(openAICompatReasoningHistoryPayload),
	}
	options := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatOpenAI,
		ResponseFormat: sdktranslator.FormatOpenAI,
		Stream:         true,
	}

	streamResult, errStream := executor.ExecuteStream(context.Background(), auth, request, options)
	if errStream != nil {
		t.Fatalf("ExecuteStream error: %v", errStream)
	}
	for chunk := range streamResult.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
	}
	if got := gjson.GetBytes(upstreamBody, "messages.1.reasoning_content").String(); got != "prior reasoning" {
		t.Fatalf("upstream reasoning_content = %q, want %q; body=%s", got, "prior reasoning", upstreamBody)
	}
}

func TestOpenAICompatExecutorPreservesReasoningContentResponsesFormat(t *testing.T) {
	payload := `{"model":"deepseek-v4-flash","input":[
		{"type":"message","role":"assistant","content":[{"type":"output_text","text":"I will look it up."}]},
		{"type":"reasoning","summary":[{"type":"summary_text","text":"prior reasoning"}]},
		{"type":"function_call","call_id":"call_1","name":"read","arguments":"{}"},
		{"type":"function_call_output","call_id":"call_1","output":"ok"},
		{"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
	]}`
	var upstreamBody []byte
	executor, server := newOpenAICompatReasoningTestExecutor(t, func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok","reasoning_content":"prior reasoning"},"finish_reason":"stop"}]}`))
	})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": server.URL,
			"api_key":  "test-key",
		},
	}
	request := cliproxyexecutor.Request{
		Model:   "deepseek-v4-flash",
		Payload: []byte(payload),
	}
	options := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatOpenAIResponse,
		ResponseFormat: sdktranslator.FormatOpenAIResponse,
	}

	if _, errExecute := executor.Execute(context.Background(), auth, request, options); errExecute != nil {
		t.Fatalf("Execute error: %v", errExecute)
	}
	found := false
	gjson.GetBytes(upstreamBody, "messages").ForEach(func(_, message gjson.Result) bool {
		if message.Get("role").String() != "assistant" || !message.Get("tool_calls").Exists() {
			return true
		}
		found = true
		if got := message.Get("reasoning_content").String(); got != "prior reasoning" {
			t.Fatalf("upstream tool-call assistant reasoning_content = %q, want %q; body=%s", got, "prior reasoning", upstreamBody)
		}
		return false
	})
	if !found {
		t.Fatalf("upstream request missing tool-call assistant message; body=%s", upstreamBody)
	}
}

func TestOpenAICompatExecutorPreservesReasoningContentResponsesFormatStreaming(t *testing.T) {
	payload := `{"model":"deepseek-v4-flash","instructions":"be brief","stream":true,"tools":[{"type":"function","name":"read","description":"read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}],"input":[
		{"type":"message","role":"assistant","content":[{"type":"output_text","text":"I will look it up."}]},
		{"type":"reasoning","summary":[{"type":"summary_text","text":"prior reasoning"}]},
		{"type":"function_call","call_id":"call_1","name":"read","arguments":"{\"path\":\"a.go\"}"},
		{"type":"function_call_output","call_id":"call_1","output":"ok"},
		{"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
	]}`
	var upstreamBody []byte
	executor, server := newOpenAICompatReasoningTestExecutor(t, func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"reasoning_content\":\"prior reasoning\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n")
	})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": server.URL,
			"api_key":  "test-key",
		},
	}
	request := cliproxyexecutor.Request{
		Model:   "deepseek-v4-flash",
		Payload: []byte(payload),
	}
	options := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatOpenAIResponse,
		ResponseFormat: sdktranslator.FormatOpenAIResponse,
		Stream:         true,
	}

	streamResult, errStream := executor.ExecuteStream(context.Background(), auth, request, options)
	if errStream != nil {
		t.Fatalf("ExecuteStream error: %v", errStream)
	}
	for chunk := range streamResult.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
	}
	found := false
	gjson.GetBytes(upstreamBody, "messages").ForEach(func(_, message gjson.Result) bool {
		if message.Get("role").String() != "assistant" || !message.Get("tool_calls").Exists() {
			return true
		}
		found = true
		if got := message.Get("reasoning_content").String(); got != "prior reasoning" {
			t.Fatalf("upstream tool-call assistant reasoning_content = %q, want %q; body=%s", got, "prior reasoning", upstreamBody)
		}
		return false
	})
	if !found {
		t.Fatalf("upstream request missing tool-call assistant message; body=%s", upstreamBody)
	}
}

func TestOpenAICompatExecutorPreservesReasoningContentResponsesFormatReasoningAfterToolOutput(t *testing.T) {
	// Some Codex thread serializations place the reasoning item after the
	// function_call_output. The tool-call assistant message must still carry
	// reasoning_content for providers (e.g. DeepSeek thinking mode) that require it.
	payload := `{"model":"deepseek-v4-flash","input":[
		{"type":"message","role":"assistant","content":[{"type":"output_text","text":"I will look it up."}]},
		{"type":"function_call","call_id":"call_1","name":"read","arguments":"{}"},
		{"type":"function_call_output","call_id":"call_1","output":"ok"},
		{"type":"reasoning","summary":[{"type":"summary_text","text":"prior reasoning"}]},
		{"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
	]}`
	var upstreamBody []byte
	executor, server := newOpenAICompatReasoningTestExecutor(t, func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok","reasoning_content":"prior reasoning"},"finish_reason":"stop"}]}`))
	})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": server.URL,
			"api_key":  "test-key",
		},
	}
	request := cliproxyexecutor.Request{
		Model:   "deepseek-v4-flash",
		Payload: []byte(payload),
	}
	options := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatOpenAIResponse,
		ResponseFormat: sdktranslator.FormatOpenAIResponse,
	}

	if _, errExecute := executor.Execute(context.Background(), auth, request, options); errExecute != nil {
		t.Fatalf("Execute error: %v", errExecute)
	}
	found := false
	gjson.GetBytes(upstreamBody, "messages").ForEach(func(_, message gjson.Result) bool {
		if message.Get("role").String() != "assistant" || !message.Get("tool_calls").Exists() {
			return true
		}
		found = true
		if got := message.Get("reasoning_content").String(); got != "prior reasoning" {
			t.Fatalf("upstream tool-call assistant reasoning_content = %q, want %q; body=%s", got, "prior reasoning", upstreamBody)
		}
		return false
	})
	if !found {
		t.Fatalf("upstream request missing tool-call assistant message; body=%s", upstreamBody)
	}
}

func TestOpenAICompatExecutorPreservesReasoningContentResponsesFormatReasoningBetweenCallAndOutput(t *testing.T) {
	payload := `{"model":"deepseek-v4-flash","input":[
		{"type":"message","role":"assistant","content":[{"type":"output_text","text":"I will look it up."}]},
		{"type":"function_call","call_id":"call_1","name":"read","arguments":"{}"},
		{"type":"reasoning","summary":[{"type":"summary_text","text":"prior reasoning"}]},
		{"type":"function_call_output","call_id":"call_1","output":"ok"},
		{"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
	]}`
	var upstreamBody []byte
	executor, server := newOpenAICompatReasoningTestExecutor(t, func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok","reasoning_content":"prior reasoning"},"finish_reason":"stop"}]}`))
	})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": server.URL,
			"api_key":  "test-key",
		},
	}
	request := cliproxyexecutor.Request{
		Model:   "deepseek-v4-flash",
		Payload: []byte(payload),
	}
	options := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatOpenAIResponse,
		ResponseFormat: sdktranslator.FormatOpenAIResponse,
	}

	if _, errExecute := executor.Execute(context.Background(), auth, request, options); errExecute != nil {
		t.Fatalf("Execute error: %v", errExecute)
	}
	found := false
	gjson.GetBytes(upstreamBody, "messages").ForEach(func(_, message gjson.Result) bool {
		if message.Get("role").String() != "assistant" || !message.Get("tool_calls").Exists() {
			return true
		}
		found = true
		if got := message.Get("reasoning_content").String(); got != "prior reasoning" {
			t.Fatalf("upstream tool-call assistant reasoning_content = %q, want %q; body=%s", got, "prior reasoning", upstreamBody)
		}
		return false
	})
	if !found {
		t.Fatalf("upstream request missing tool-call assistant message; body=%s", upstreamBody)
	}
}

func TestOpenAICompatExecutorUsesCompatibleClaudeTranslation(t *testing.T) {
	var upstreamBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer server.Close()

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	auth := &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url": server.URL,
			"api_key":  "test-key",
		},
	}
	request := cliproxyexecutor.Request{
		Model:   "deepseek-v4-flash",
		Payload: []byte(`{"model":"deepseek-v4-flash","messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"prior reasoning","signature":""},{"type":"tool_use","id":"call_1","name":"Read","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"ok"}]}]}`),
		Metadata: map[string]any{
			"cliproxy.resolved_api_key_model_info": &registry.ModelInfo{IsCompat: true},
		},
	}
	options := cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatClaude,
		ResponseFormat: sdktranslator.FormatOpenAI,
	}

	if _, errExecute := executor.Execute(context.Background(), auth, request, options); errExecute != nil {
		t.Fatalf("Execute error: %v", errExecute)
	}

	assistant := gjson.GetBytes(upstreamBody, "messages.0")
	if got := assistant.Get("reasoning_content").String(); got != "prior reasoning" {
		t.Fatalf("reasoning_content = %q, want %q; body=%s", got, "prior reasoning", upstreamBody)
	}
	if !assistant.Get("tool_calls").Exists() {
		t.Fatalf("tool_calls missing from upstream request: %s", upstreamBody)
	}
}
