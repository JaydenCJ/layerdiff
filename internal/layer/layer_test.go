// Tests for layer-key derivation and natural name ordering — the two
// pieces that make per-layer reports line up with how humans read
// transformer checkpoints.
package layer

import (
	"reflect"
	"testing"
)

func TestKeyAutoStopsAtFirstIntegerSegment(t *testing.T) {
	cases := map[string]string{
		"model.layers.17.attn.wq.weight":     "model.layers.17",
		"transformer.h.3.mlp.fc.bias":        "transformer.h.3",
		"model.layers.2.experts.5.w1.weight": "model.layers.2", // first int wins, not the expert index
		"decoder.block.0.norm.weight":        "decoder.block.0",
	}
	for name, want := range cases {
		if got := Key(name, 0); got != want {
			t.Errorf("Key(%q) = %q, want %q", name, got, want)
		}
	}
}

func TestKeyAutoWithoutIntegerDropsParameterName(t *testing.T) {
	cases := map[string]string{
		"model.embed_tokens.weight": "model.embed_tokens",
		"model.norm.weight":         "model.norm",
		"lm_head.weight":            "lm_head",
	}
	for name, want := range cases {
		if got := Key(name, 0); got != want {
			t.Errorf("Key(%q) = %q, want %q", name, got, want)
		}
	}
	if got := Key("bias", 0); got != "bias" {
		t.Errorf("single-segment name must be its own group, got %q", got)
	}
}

func TestKeyExplicitDepth(t *testing.T) {
	if got := Key("model.layers.17.attn.wq.weight", 2); got != "model.layers" {
		t.Errorf("depth 2: got %q", got)
	}
	if got := Key("model.layers.17.attn.wq.weight", 4); got != "model.layers.17.attn" {
		t.Errorf("depth 4: got %q", got)
	}
	if got := Key("a.b", 10); got != "a.b" {
		t.Errorf("over-deep key must be the whole name, got %q", got)
	}
}

func TestNaturalLessComparesIntegerSegmentsNumerically(t *testing.T) {
	if !NaturalLess("model.layers.2.w", "model.layers.10.w") {
		t.Fatal("layer 2 must sort before layer 10")
	}
	if NaturalLess("model.layers.10.w", "model.layers.2.w") {
		t.Fatal("layer 10 must not sort before layer 2")
	}
	// Prefix relationship: shorter name first.
	if !NaturalLess("model.layers", "model.layers.0") {
		t.Fatal("prefix must sort first")
	}
}

func TestSortNamesNaturalOrder(t *testing.T) {
	names := []string{
		"model.layers.10.weight",
		"model.embed_tokens.weight",
		"model.layers.2.weight",
		"model.layers.1.weight",
	}
	SortNames(names)
	want := []string{
		"model.embed_tokens.weight",
		"model.layers.1.weight",
		"model.layers.2.weight",
		"model.layers.10.weight",
	}
	if !reflect.DeepEqual(names, want) {
		t.Fatalf("got %v, want %v", names, want)
	}
}
