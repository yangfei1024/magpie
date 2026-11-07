package gateway

import (
	"bytes"
	"encoding/json"
	"strconv"
)

// xAI's Messages stream never says which content block a delta belongs
// to: of a real grok-4.7 turn captured at the gateway (2026-11-07), all
// ten content_block_delta events went out bare —
//
//	data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"…"}}
//
// — where Anthropic's own stream carries the block's index on every
// content_block_start, _delta and _stop alike. A client that validates
// the events as they come turns the whole turn away: ZCode's decoder
// answers ["index"]: expected number, received undefined, and the turn
// fails after the model has spoken (#3392's family, the reply side of
// xAI's strict-but-untidy API). The start and stop events do carry their
// index, so the block a bare delta belongs to is the one the stream is
// in: xaiIndexTidy says it where the wire left it out, an index that is
// there stays as it came and moves the stream into its block, and a line
// without a content_block event in it passes untouched.

var contentBlockEvent = []byte("content_block_")

type xaiIndexTidy struct {
	sse bool
	buf []byte
	// cur is the index of the block the stream is inside: the last
	// content_block_start's index, and the first block's 0 before any
	cur int
}

func (t *xaiIndexTidy) write(b []byte) []byte {
	t.buf = append(t.buf, b...)
	if !t.sse {
		return nil
	}
	i := bytes.LastIndexByte(t.buf, '\n')
	if i < 0 {
		return nil
	}
	out := t.lines(t.buf[:i+1])
	t.buf = append(t.buf[:0], t.buf[i+1:]...)
	return out
}

func (t *xaiIndexTidy) flush() []byte {
	b := t.buf
	t.buf = nil
	return t.lines(b)
}

func (t *xaiIndexTidy) lines(b []byte) []byte {
	if !bytes.Contains(b, contentBlockEvent) {
		return append([]byte(nil), b...)
	}
	var out []byte
	for len(b) > 0 {
		line := b
		if i := bytes.IndexByte(b, '\n'); i >= 0 {
			line, b = b[:i+1], b[i+1:]
		} else {
			b = nil
		}
		body := bytes.TrimRight(line, "\r\n")
		data, ok := bytes.CutPrefix(body, []byte("data:"))
		if !ok {
			out = append(out, line...)
			continue
		}
		if nb, changed := t.event(bytes.TrimSpace(data)); changed {
			out = append(append(append(out, "data: "...), nb...), line[len(body):]...)
			continue
		}
		out = append(out, line...)
	}
	return out
}

// event gives a content_block event its index where the wire left it out,
// and whether it did: the block a stream is inside is the last start's
// index, one a start itself carries, and one a delta or a stop that
// carries moves the stream into. An event that no longer needs one, or
// whose index the wire already says, goes back unchanged.
func (t *xaiIndexTidy) event(data []byte) ([]byte, bool) {
	var ev map[string]json.RawMessage
	if json.Unmarshal(data, &ev) != nil {
		return nil, false
	}
	var typ string
	if json.Unmarshal(ev["type"], &typ) != nil {
		return nil, false
	}
	switch typ {
	case "content_block_start", "content_block_delta", "content_block_stop":
	default:
		return nil, false
	}
	if raw, there := ev["index"]; there {
		if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			delete(ev, "index")
		} else {
			var idx float64
			if json.Unmarshal(raw, &idx) == nil {
				t.cur = int(idx)
			}
			return nil, false
		}
	}
	ev["index"] = json.RawMessage(strconv.Itoa(t.cur))
	nb, err := marshalPlain(ev)
	return nb, err == nil
}
