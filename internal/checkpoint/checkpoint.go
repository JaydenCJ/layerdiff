// Package checkpoint resolves a user-supplied path into a uniform view
// of a model checkpoint: a set of named tensors, each addressable as a
// bounded byte stream. Three layouts are supported:
//
//   - a single .safetensors file,
//   - a sharded checkpoint described by a *.safetensors.index.json
//     weight map (the layout produced for large models),
//   - a directory, which resolves to the index inside it if there is
//     one, or to the union of its loose .safetensors files otherwise.
//
// Tensor data is never loaded whole: Reader returns a section of the
// underlying shard file, opened on demand and closed by the caller.
package checkpoint

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/JaydenCJ/layerdiff/internal/layer"
	"github.com/JaydenCJ/layerdiff/internal/safetensors"
)

// Kind labels how a checkpoint was resolved.
const (
	KindFile      = "safetensors"
	KindSharded   = "sharded"
	KindDirectory = "directory"
)

// Tensor is one tensor of a checkpoint, addressable for streaming.
type Tensor struct {
	Name  string
	DType string
	Shape []int64
	// Bytes is the on-disk data size; Elems the element count implied by
	// the shape (a rank-0 tensor is one scalar element).
	Bytes int64
	Elems int64

	path       string
	begin, end int64
}

// Reader opens the shard file and returns a reader over exactly this
// tensor's bytes. Each call opens its own file handle; Close releases it.
func (t *Tensor) Reader() (io.ReadCloser, error) {
	f, err := os.Open(t.path)
	if err != nil {
		return nil, err
	}
	return &sectionCloser{
		SectionReader: io.NewSectionReader(f, t.begin, t.end-t.begin),
		file:          f,
	}, nil
}

type sectionCloser struct {
	*io.SectionReader
	file *os.File
}

func (s *sectionCloser) Close() error { return s.file.Close() }

// Checkpoint is the resolved, header-only view of a model checkpoint.
type Checkpoint struct {
	// Path is what the user asked to open; Kind how it resolved.
	Path string
	Kind string
	// Shards lists the underlying safetensors file names (base names,
	// sorted), one entry for a single-file checkpoint.
	Shards []string
	// Metadata carries the __metadata__ table of a single file, or the
	// stringified "metadata" object of a sharded index.
	Metadata map[string]string

	byName map[string]*Tensor
	names  []string
}

// Names returns all tensor names in natural order.
func (c *Checkpoint) Names() []string { return c.names }

// Tensor looks a tensor up by name.
func (c *Checkpoint) Tensor(name string) (*Tensor, bool) {
	t, ok := c.byName[name]
	return t, ok
}

// Len returns the number of tensors.
func (c *Checkpoint) Len() int { return len(c.names) }

// DataBytes returns the total tensor data size across all shards.
func (c *Checkpoint) DataBytes() int64 {
	var n int64
	for _, t := range c.byName {
		n += t.Bytes
	}
	return n
}

// Open resolves path — file, index, or directory — into a Checkpoint.
func Open(path string) (*Checkpoint, error) {
	st, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if st.IsDir() {
		return openDirectory(path)
	}
	if strings.HasSuffix(path, ".json") {
		return openIndex(path)
	}
	return openSingle(path)
}

func openSingle(path string) (*Checkpoint, error) {
	sf, err := safetensors.Open(path)
	if err != nil {
		return nil, err
	}
	cp := &Checkpoint{
		Path:     path,
		Kind:     KindFile,
		Shards:   []string{filepath.Base(path)},
		Metadata: sf.Metadata,
		byName:   make(map[string]*Tensor, len(sf.Tensors)),
	}
	if err := cp.addFile(sf); err != nil {
		return nil, err
	}
	cp.finish()
	return cp, nil
}

func openDirectory(dir string) (*Checkpoint, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var indexes, shards []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		switch {
		case strings.HasSuffix(name, ".safetensors.index.json"):
			indexes = append(indexes, name)
		case strings.HasSuffix(name, ".safetensors"):
			shards = append(shards, name)
		}
	}
	if len(indexes) > 1 {
		return nil, fmt.Errorf("%s: %d index files found, pass one explicitly", dir, len(indexes))
	}
	if len(indexes) == 1 {
		cp, err := openIndex(filepath.Join(dir, indexes[0]))
		if err != nil {
			return nil, err
		}
		cp.Path = dir // label reports with what the user passed
		return cp, nil
	}
	if len(shards) == 0 {
		return nil, fmt.Errorf("%s: no .safetensors files in directory", dir)
	}
	layer.SortNames(shards)

	cp := &Checkpoint{
		Path:   dir,
		Kind:   KindDirectory,
		Shards: shards,
		byName: make(map[string]*Tensor),
	}
	for _, name := range shards {
		sf, err := safetensors.Open(filepath.Join(dir, name))
		if err != nil {
			return nil, err
		}
		if err := cp.addFile(sf); err != nil {
			return nil, err
		}
		// A directory of loose shards has no canonical metadata; keep the
		// first table seen so `ls` still surfaces provenance keys.
		if cp.Metadata == nil {
			cp.Metadata = sf.Metadata
		}
	}
	if len(shards) == 1 {
		cp.Kind = KindFile
	}
	cp.finish()
	return cp, nil
}

// addFile merges one parsed safetensors file into the checkpoint,
// rejecting duplicate tensor names across shards.
func (c *Checkpoint) addFile(sf *safetensors.File) error {
	for _, t := range sf.Tensors {
		if prev, dup := c.byName[t.Name]; dup {
			return fmt.Errorf("tensor %q appears in both %s and %s",
				t.Name, filepath.Base(prev.path), filepath.Base(sf.Path))
		}
		c.byName[t.Name] = &Tensor{
			Name:  t.Name,
			DType: t.DType,
			Shape: t.Shape,
			Bytes: t.Bytes(),
			Elems: t.Elems(),
			path:  sf.Path,
			begin: t.Begin,
			end:   t.End,
		}
	}
	return nil
}

func (c *Checkpoint) finish() {
	c.names = make([]string, 0, len(c.byName))
	for name := range c.byName {
		c.names = append(c.names, name)
	}
	layer.SortNames(c.names)
}
