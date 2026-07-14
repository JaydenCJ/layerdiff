// Package stwrite is a minimal safetensors writer. layerdiff itself only
// ever reads checkpoints; this writer exists so tests, examples, and the
// smoke script can fabricate real checkpoint files deterministically,
// without any ML framework.
package stwrite

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"

	"github.com/JaydenCJ/layerdiff/internal/dtype"
)

// Tensor is one tensor to serialize: a name, a safetensors dtype string,
// a shape, and the raw little-endian element bytes.
type Tensor struct {
	Name  string
	DType string
	Shape []int64
	Data  []byte
}

// Encode builds a complete safetensors file image. Tensors are laid out
// in the order given; the header is padded with spaces to an 8-byte
// boundary, matching the reference implementation's convention.
func Encode(metadata map[string]string, tensors []Tensor) ([]byte, error) {
	header := make(map[string]any, len(tensors)+1)
	if metadata != nil {
		header["__metadata__"] = metadata
	}
	offset := int64(0)
	for _, t := range tensors {
		if _, dup := header[t.Name]; dup {
			return nil, fmt.Errorf("stwrite: duplicate tensor %q", t.Name)
		}
		if info := dtype.Lookup(t.DType); info.Known() {
			elems := int64(1)
			for _, d := range t.Shape {
				elems *= d
			}
			if want := elems * int64(info.Size); want != int64(len(t.Data)) {
				return nil, fmt.Errorf("stwrite: tensor %q: %s shape %v needs %d bytes, got %d",
					t.Name, t.DType, t.Shape, want, len(t.Data))
			}
		}
		shape := t.Shape
		if shape == nil {
			shape = []int64{}
		}
		header[t.Name] = map[string]any{
			"dtype":        t.DType,
			"shape":        shape,
			"data_offsets": []int64{offset, offset + int64(len(t.Data))},
		}
		offset += int64(len(t.Data))
	}

	js, err := json.Marshal(header)
	if err != nil {
		return nil, err
	}
	if pad := (8 - len(js)%8) % 8; pad > 0 {
		js = append(js, bytes.Repeat([]byte(" "), pad)...)
	}

	var out bytes.Buffer
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(js)))
	out.Write(lenBuf[:])
	out.Write(js)
	for _, t := range tensors {
		out.Write(t.Data)
	}
	return out.Bytes(), nil
}

// WriteFile encodes and writes a safetensors file at path.
func WriteFile(path string, metadata map[string]string, tensors []Tensor) error {
	b, err := Encode(metadata, tensors)
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// F32 packs float32 values as little-endian bytes.
func F32(vals ...float32) []byte {
	out := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(v))
	}
	return out
}

// F64 packs float64 values as little-endian bytes.
func F64(vals ...float64) []byte {
	out := make([]byte, 8*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint64(out[8*i:], math.Float64bits(v))
	}
	return out
}

// U16 packs raw 16-bit payloads as little-endian bytes; use it to build
// F16 and BF16 tensors from explicit bit patterns.
func U16(vals ...uint16) []byte {
	out := make([]byte, 2*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint16(out[2*i:], v)
	}
	return out
}

// I64 packs int64 values as little-endian bytes.
func I64(vals ...int64) []byte {
	out := make([]byte, 8*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint64(out[8*i:], uint64(v))
	}
	return out
}

// I8 packs int8 values as bytes.
func I8(vals ...int8) []byte {
	out := make([]byte, len(vals))
	for i, v := range vals {
		out[i] = byte(v)
	}
	return out
}
