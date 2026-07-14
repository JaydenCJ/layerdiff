// Tests for tensor-name filtering.
package diff

import "testing"

func TestNilFilterMatchesEverything(t *testing.T) {
	f, err := NewFilter(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if f != nil {
		t.Fatal("empty patterns must yield a nil (match-all) filter")
	}
	if !f.Match("anything.at.all") {
		t.Fatal("nil filter must match")
	}
}

func TestIncludeRestrictsExcludeWins(t *testing.T) {
	f, err := NewFilter([]string{"model.layers.*"}, []string{"*.bias"})
	if err != nil {
		t.Fatal(err)
	}
	if !f.Match("model.layers.3.attn.weight") {
		t.Fatal("included name must match")
	}
	if f.Match("lm_head.weight") {
		t.Fatal("name outside include must not match")
	}
	if f.Match("model.layers.3.attn.bias") {
		t.Fatal("exclude must override include")
	}
}

func TestExcludeOnlyFilter(t *testing.T) {
	f, err := NewFilter(nil, []string{"model.rotary.*"})
	if err != nil {
		t.Fatal(err)
	}
	if !f.Match("model.layers.0.w") || f.Match("model.rotary.inv_freq") {
		t.Fatal("exclude-only filter wrong")
	}
}

func TestInvalidPatternRejectedUpFront(t *testing.T) {
	if _, err := NewFilter([]string{"[unclosed"}, nil); err == nil {
		t.Fatal("malformed pattern must be a construction error")
	}
}
