// Package safetensors parses the header of a .safetensors file: an
// 8-byte little-endian length, a JSON table mapping tensor names to
// dtype/shape/offsets, then the raw tensor data. Only the header is ever
// held in memory; tensor bytes are addressed by absolute file offsets so
// callers can stream them.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"github.com/JaydenCJ/layerdiff/internal/dtype"
)

// MaxHeaderSize caps the JSON header a file may declare. Real headers
// are a few MiB at most even for thousand-tensor checkpoints; the cap
// keeps a corrupt or hostile length prefix from forcing a huge
// allocation.
const MaxHeaderSize = 100 << 20

// Tensor is one entry of the header table, with offsets already
// converted to absolute positions within the file.
type Tensor struct {
	Name  string
	DType string
	Shape []int64
	// Begin and End are absolute byte offsets in the file; the tensor's
	// raw little-endian data is file[Begin:End].
	Begin int64
	End   int64
}

// Bytes returns the on-disk size of the tensor data.
func (t Tensor) Bytes() int64 { return t.End - t.Begin }

// Elems returns the element count implied by the shape. A rank-0 shape
// is a scalar (one element); any zero dimension makes the tensor empty.
func (t Tensor) Elems() int64 {
	n := int64(1)
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// File is the parsed header of one safetensors file.
type File struct {
	Path     string
	Size     int64
	Metadata map[string]string
	// Tensors is sorted by name.
	Tensors []Tensor
	// DataStart is the absolute offset where tensor data begins
	// (8 + header length).
	DataStart int64
}

// headerEntry mirrors one JSON value in the header table.
type headerEntry struct {
	DType       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// Open reads and validates the header of the safetensors file at path.
// The file handle is closed before returning; tensor data is read later
// through offsets.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	return parse(f, path, st.Size())
}

func parse(r io.Reader, path string, size int64) (*File, error) {
	var lenBuf [8]byte
	if _, err := io.ReadFull(r, lenBuf[:]); err != nil {
		return nil, fmt.Errorf("%s: not a safetensors file: %w", path, err)
	}
	headerLen := binary.LittleEndian.Uint64(lenBuf[:])
	if headerLen > MaxHeaderSize {
		return nil, fmt.Errorf("%s: header length %d exceeds the %d-byte cap", path, headerLen, int64(MaxHeaderSize))
	}
	dataStart := int64(8) + int64(headerLen)
	if dataStart > size {
		return nil, fmt.Errorf("%s: header length %d overruns the %d-byte file", path, headerLen, size)
	}

	raw := make([]byte, headerLen)
	if _, err := io.ReadFull(r, raw); err != nil {
		return nil, fmt.Errorf("%s: truncated header: %w", path, err)
	}

	var table map[string]json.RawMessage
	if err := json.Unmarshal(raw, &table); err != nil {
		return nil, fmt.Errorf("%s: header is not a JSON object: %w", path, err)
	}

	out := &File{Path: path, Size: size, DataStart: dataStart}
	dataLen := size - dataStart
	for name, msg := range table {
		if name == "__metadata__" {
			if err := json.Unmarshal(msg, &out.Metadata); err != nil {
				return nil, fmt.Errorf("%s: __metadata__ must map strings to strings: %w", path, err)
			}
			continue
		}
		var e headerEntry
		if err := json.Unmarshal(msg, &e); err != nil {
			return nil, fmt.Errorf("%s: tensor %q: malformed header entry: %w", path, name, err)
		}
		t, err := validateEntry(name, e, dataLen)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		t.Begin += dataStart
		t.End += dataStart
		out.Tensors = append(out.Tensors, t)
	}

	sort.Slice(out.Tensors, func(i, j int) bool { return out.Tensors[i].Name < out.Tensors[j].Name })
	return out, nil
}

// validateEntry checks one header entry against the data region length.
// The returned tensor still carries data-relative offsets.
func validateEntry(name string, e headerEntry, dataLen int64) (Tensor, error) {
	if e.DType == "" {
		return Tensor{}, fmt.Errorf("tensor %q: missing dtype", name)
	}
	elems := int64(1)
	for _, d := range e.Shape {
		if d < 0 {
			return Tensor{}, fmt.Errorf("tensor %q: negative dimension %d", name, d)
		}
		if d != 0 && elems > math.MaxInt64/d {
			return Tensor{}, fmt.Errorf("tensor %q: shape overflows int64", name)
		}
		elems *= d
	}
	begin, end := e.DataOffsets[0], e.DataOffsets[1]
	if begin < 0 || end < begin {
		return Tensor{}, fmt.Errorf("tensor %q: invalid data_offsets [%d, %d]", name, begin, end)
	}
	if end > dataLen {
		return Tensor{}, fmt.Errorf("tensor %q: data_offsets end %d overruns the %d-byte data region", name, end, dataLen)
	}
	if info := dtype.Lookup(e.DType); info.Known() {
		want := elems * int64(info.Size)
		if got := end - begin; got != want {
			return Tensor{}, fmt.Errorf("tensor %q: %s shape %v needs %d bytes, data_offsets span %d",
				name, e.DType, e.Shape, want, got)
		}
	}
	shape := make([]int64, len(e.Shape))
	copy(shape, e.Shape)
	return Tensor{Name: name, DType: e.DType, Shape: shape, Begin: begin, End: end}, nil
}
