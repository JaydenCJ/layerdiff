// Package layer maps tensor names onto layer groups and provides the
// natural ordering used everywhere names are listed, so that
// "model.layers.10" sorts after "model.layers.2".
package layer

import (
	"sort"
	"strconv"
	"strings"
)

// Key derives the layer group for a dot-separated tensor name.
//
// With depth > 0 the key is simply the first depth segments.
//
// With depth == 0 (auto) the heuristic mirrors how transformer
// checkpoints are named: the key runs up to and including the first
// integer segment ("model.layers.17.attn.wq.weight" → "model.layers.17",
// "transformer.h.3.mlp.fc.bias" → "transformer.h.3"). Names without an
// integer segment drop their final segment — the parameter name — so
// "model.embed_tokens.weight" groups as "model.embed_tokens"; a
// single-segment name is its own group.
func Key(name string, depth int) string {
	segs := strings.Split(name, ".")
	if depth > 0 {
		if depth >= len(segs) {
			return name
		}
		return strings.Join(segs[:depth], ".")
	}
	for i, s := range segs {
		if isInt(s) {
			return strings.Join(segs[:i+1], ".")
		}
	}
	if len(segs) > 1 {
		return strings.Join(segs[:len(segs)-1], ".")
	}
	return name
}

func isInt(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// NaturalLess orders dot-separated names segment by segment, comparing
// all-digit segments numerically so layer 2 precedes layer 10.
func NaturalLess(a, b string) bool {
	as, bs := strings.Split(a, "."), strings.Split(b, ".")
	for i := 0; i < len(as) && i < len(bs); i++ {
		x, y := as[i], bs[i]
		if x == y {
			continue
		}
		xi, xerr := strconv.ParseInt(x, 10, 64)
		yi, yerr := strconv.ParseInt(y, 10, 64)
		if xerr == nil && yerr == nil && xi != yi {
			return xi < yi
		}
		return x < y
	}
	return len(as) < len(bs)
}

// SortNames sorts names in natural order, in place.
func SortNames(names []string) {
	sort.Slice(names, func(i, j int) bool { return NaturalLess(names[i], names[j]) })
}
