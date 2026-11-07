package gateway

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// xAI's real grok-4.7 stream, captured at the gateway 2026-11-07 (the
// turn ZCode's strict decoder turned away): every content_block_delta
// bare of its index, the starts and the stops carrying theirs — and
// xAI's own numbering reusing 0 for the second block, which the tidy
// follows as the wire says it.
const xaiBareDeltaStream = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"grok-4.7","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"The user sent a greeting."}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}

event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"!"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

`

// TestXAIStreamDeltasCarryTheBlocksIndex relays the captured stream and
// asks that every delta the client sees says its block, the framed events
// going as they came.
func TestXAIStreamDeltasCarryTheBlocksIndex(t *testing.T) {
	f := &fake{t: t, reply: xaiBareDeltaStream}
	up := setup(t, provider.Anthropic, f)
	if err := provider.Save(provider.Provider{ID: "fake", Name: "Fake", Key: "k", Models: []string{"grok-4.7"}, Anthropic: "http://api.x.ai"}); err != nil {
		t.Fatal(err)
	}
	s := New()
	addr := strings.TrimPrefix(up.URL, "http://")
	s.client = &http.Client{Transport: &http.Transport{DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, addr)
	}}}
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"fake/grok-4.7","stream":true,"max_tokens":64,"messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]}]}`)))
	if rec.Code != 200 {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	client := rec.Body.String()
	deltas := 0
	for _, line := range strings.Split(client, "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || !strings.Contains(data, `"content_block_delta"`) {
			continue
		}
		var ev struct {
			Index *float64 `json:"index"`
			Delta struct {
				Type string `json:"type"`
			} `json:"delta"`
		}
		if json.Unmarshal([]byte(data), &ev) != nil {
			continue
		}
		if ev.Index == nil {
			t.Errorf("a %s delta reached the client without its index: %s", ev.Delta.Type, data)
		}
		deltas++
	}
	if deltas != 3 {
		t.Errorf("3 deltas went upstream, %d came out", deltas)
	}
	// the framed events keep their index, the thinking block gaining the
	// signature a Messages client reads as required (thinking_signature.go)
	for _, want := range []string{
		`"signature":""`,
		`{"type":"content_block_stop","index":0}`,
		`event: message_start`,
	} {
		if !strings.Contains(client, want) {
			t.Errorf("a framed event was rewritten: %s", want)
		}
	}
}

// TestXaiIndexTidyLines walks the tidy itself: an index left out or null
// is said, one that is there stays and moves the stream into its block,
// and lines without a content_block event pass untouched, cut across
// writes as a stream's chunks land.
func TestXaiIndexTidyLines(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"a bare delta is told its block",
			`data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"hi"}}` + "\n\n",
			`data: {"delta":{"type":"thinking_delta","thinking":"hi"},"index":0,"type":"content_block_delta"}` + "\n\n"},
		{"an index the wire says stays",
			`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hi"}}` + "\n\n",
			`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hi"}}` + "\n\n"},
		{"an index that follows moves the stream into its block",
			`data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}` + "\n" +
				`data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}` + "\n\n",
			`data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}` + "\n" +
				`data: {"delta":{"type":"text_delta","text":"hi"},"index":2,"type":"content_block_delta"}` + "\n\n"},
		{"a null index is said as none",
			`data: {"type":"content_block_delta","index":null,"delta":{"type":"text_delta","text":"hi"}}` + "\n\n",
			`data: {"delta":{"type":"text_delta","text":"hi"},"index":0,"type":"content_block_delta"}` + "\n\n"},
		{"no content_block event in the line, untouched",
			"data: {\"type\":\"ping\"}\n\nevent: ping\n\n",
			"data: {\"type\":\"ping\"}\n\nevent: ping\n\n"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tidy := &xaiIndexTidy{sse: true}
			got := tidy.write([]byte(c.in))
			if rest := tidy.flush(); len(rest) > 0 {
				got = append(got, rest...)
			}
			if string(got) != c.want {
				t.Errorf("got  %q\nwant %q", got, c.want)
			}
		})
	}
	// a stream cut mid-event holds back until the rest arrives
	tidy := &xaiIndexTidy{sse: true}
	half := tidy.write([]byte(`data: {"type":"content_block_delta","del`))
	if half != nil {
		t.Errorf("a half event went out: %q", half)
	}
	rest := tidy.write([]byte(`ta":{"type":"text_delta","text":"hi"}}` + "\n\n"))
	if !strings.Contains(string(rest), `"index":0`) {
		t.Errorf("the reassembled event lost its fill: %q", rest)
	}
}
