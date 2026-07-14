// Tensor-name filtering for diff and ls: shell-style patterns matched
// against the full dot-separated name. Since tensor names contain no
// path separators, `*` effectively matches any run of characters, so
// "model.layers.7.*" or "*.bias" behave the way users expect.
package diff

import (
	"fmt"
	"path"
)

// Filter selects tensor names by include/exclude patterns. A nil Filter
// matches everything.
type Filter struct {
	include []string
	exclude []string
}

// NewFilter validates the patterns up front so a typo like "[q" is a
// usage error, not a silent zero-match. Returns nil when both lists are
// empty.
func NewFilter(include, exclude []string) (*Filter, error) {
	if len(include) == 0 && len(exclude) == 0 {
		return nil, nil
	}
	for _, p := range append(append([]string{}, include...), exclude...) {
		if _, err := path.Match(p, "probe"); err != nil {
			return nil, fmt.Errorf("invalid pattern %q", p)
		}
	}
	return &Filter{include: include, exclude: exclude}, nil
}

// Match reports whether name passes the filter: it must match at least
// one include pattern (if any are given) and no exclude pattern.
func (f *Filter) Match(name string) bool {
	if f == nil {
		return true
	}
	if len(f.include) > 0 {
		hit := false
		for _, p := range f.include {
			if ok, _ := path.Match(p, name); ok {
				hit = true
				break
			}
		}
		if !hit {
			return false
		}
	}
	for _, p := range f.exclude {
		if ok, _ := path.Match(p, name); ok {
			return false
		}
	}
	return true
}
