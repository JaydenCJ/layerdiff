// Package dtype describes the tensor element types found in safetensors
// checkpoints: their on-disk size and, where meaningful, a decoder from
// little-endian bytes to float64 so statistics can be streamed without
// materializing a tensor.
//
// Types layerdiff cannot decode numerically (for example the FP8
// variants) are still first-class citizens: their size is known, so
// shapes validate and byte-level hashing and comparison work; only the
// numeric statistics are skipped. Completely unknown type names degrade
// further to "opaque": hash-and-compare only, no shape validation.
package dtype

import (
	"encoding/binary"
	"math"
)

// Info describes one element type as named in a safetensors header.
type Info struct {
	// Name is the safetensors dtype string, e.g. "F32" or "BF16".
	Name string
	// Size is the number of bytes per element, or 0 when the type is
	// unknown to layerdiff.
	Size int
	// Decode converts one little-endian element to float64, or is nil
	// when the type has a known size but no numeric interpretation
	// layerdiff implements (e.g. FP8).
	Decode func(b []byte) float64
}

// Known reports whether the element size is known, i.e. shapes can be
// validated against byte lengths.
func (i Info) Known() bool { return i.Size > 0 }

// Numeric reports whether elements can be decoded to float64 for
// statistics and elementwise comparison.
func (i Info) Numeric() bool { return i.Decode != nil }

// Lookup returns the Info for a safetensors dtype name. Unknown names
// yield an opaque Info (Size 0, no decoder) rather than an error, so a
// checkpoint written by a newer toolchain still hashes and diffs.
func Lookup(name string) Info {
	if in, ok := registry[name]; ok {
		return in
	}
	return Info{Name: name}
}

var registry = map[string]Info{
	"F64":  {Name: "F64", Size: 8, Decode: decodeF64},
	"F32":  {Name: "F32", Size: 4, Decode: decodeF32},
	"F16":  {Name: "F16", Size: 2, Decode: decodeF16},
	"BF16": {Name: "BF16", Size: 2, Decode: decodeBF16},
	"I64":  {Name: "I64", Size: 8, Decode: decodeI64},
	"I32":  {Name: "I32", Size: 4, Decode: decodeI32},
	"I16":  {Name: "I16", Size: 2, Decode: decodeI16},
	"I8":   {Name: "I8", Size: 1, Decode: decodeI8},
	"U64":  {Name: "U64", Size: 8, Decode: decodeU64},
	"U32":  {Name: "U32", Size: 4, Decode: decodeU32},
	"U16":  {Name: "U16", Size: 2, Decode: decodeU16},
	"U8":   {Name: "U8", Size: 1, Decode: decodeU8},
	"BOOL": {Name: "BOOL", Size: 1, Decode: decodeBool},
	// FP8 formats: sized (so shapes validate and bytes hash) but not
	// decoded — their per-format bit semantics are out of scope for 0.1.0.
	"F8_E4M3": {Name: "F8_E4M3", Size: 1},
	"F8_E5M2": {Name: "F8_E5M2", Size: 1},
}

func decodeF64(b []byte) float64 {
	return math.Float64frombits(binary.LittleEndian.Uint64(b))
}

func decodeF32(b []byte) float64 {
	return float64(math.Float32frombits(binary.LittleEndian.Uint32(b)))
}

// decodeF16 expands an IEEE 754 binary16 value exactly: every half float
// (normal, subnormal, ±0, ±Inf, NaN) is representable in float64.
func decodeF16(b []byte) float64 {
	u := binary.LittleEndian.Uint16(b)
	sign := u >> 15
	exp := (u >> 10) & 0x1f
	frac := u & 0x3ff
	var v float64
	switch {
	case exp == 0:
		// Zero or subnormal: value = frac × 2⁻²⁴.
		v = math.Ldexp(float64(frac), -24)
	case exp == 0x1f:
		if frac != 0 {
			return math.NaN()
		}
		v = math.Inf(1)
	default:
		// Normal: value = (1 + frac/1024) × 2^(exp−15).
		v = math.Ldexp(1+float64(frac)/1024, int(exp)-15)
	}
	if sign == 1 {
		v = -v
	}
	return v
}

// decodeBF16 widens bfloat16 to float32 by left-shifting into the high
// bits — bfloat16 is exactly the top half of an IEEE binary32.
func decodeBF16(b []byte) float64 {
	u := binary.LittleEndian.Uint16(b)
	return float64(math.Float32frombits(uint32(u) << 16))
}

// Integer decoders. Values beyond 2⁵³ lose precision in float64; that
// only affects displayed statistics, never hashing or byte comparison.
func decodeI64(b []byte) float64 { return float64(int64(binary.LittleEndian.Uint64(b))) }
func decodeI32(b []byte) float64 { return float64(int32(binary.LittleEndian.Uint32(b))) }
func decodeI16(b []byte) float64 { return float64(int16(binary.LittleEndian.Uint16(b))) }
func decodeI8(b []byte) float64  { return float64(int8(b[0])) }
func decodeU64(b []byte) float64 { return float64(binary.LittleEndian.Uint64(b)) }
func decodeU32(b []byte) float64 { return float64(binary.LittleEndian.Uint32(b)) }
func decodeU16(b []byte) float64 { return float64(binary.LittleEndian.Uint16(b)) }
func decodeU8(b []byte) float64  { return float64(b[0]) }

func decodeBool(b []byte) float64 {
	if b[0] != 0 {
		return 1
	}
	return 0
}
