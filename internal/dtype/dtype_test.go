// Tests for the dtype registry and the little-endian element decoders.
// The half-precision cases matter most: F16 subnormals and BF16 specials
// are exactly where a sloppy decoder silently corrupts statistics.
package dtype

import (
	"math"
	"testing"
)

func TestLookupKnownSizes(t *testing.T) {
	want := map[string]int{
		"F64": 8, "F32": 4, "F16": 2, "BF16": 2,
		"I64": 8, "I32": 4, "I16": 2, "I8": 1,
		"U64": 8, "U32": 4, "U16": 2, "U8": 1,
		"BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
	}
	for name, size := range want {
		info := Lookup(name)
		if !info.Known() || info.Size != size {
			t.Errorf("Lookup(%q): size = %d, want %d", name, info.Size, size)
		}
	}
}

func TestLookupUnknownNameIsOpaque(t *testing.T) {
	info := Lookup("F4_SOMEDAY")
	if info.Known() || info.Numeric() {
		t.Fatalf("unknown dtype must be opaque, got %+v", info)
	}
	if info.Name != "F4_SOMEDAY" {
		t.Fatalf("opaque Info must keep the name, got %q", info.Name)
	}
}

func TestFP8IsSizedButNotNumeric(t *testing.T) {
	// FP8 tensors must still hash and byte-compare (Known), but must not
	// feed garbage into statistics (not Numeric).
	for _, name := range []string{"F8_E4M3", "F8_E5M2"} {
		info := Lookup(name)
		if !info.Known() || info.Numeric() {
			t.Errorf("%s: want sized-but-not-numeric, got Known=%v Numeric=%v",
				name, info.Known(), info.Numeric())
		}
	}
}

func TestDecodeF32AndF64(t *testing.T) {
	f32 := Lookup("F32").Decode
	if got := f32([]byte{0x00, 0x00, 0x80, 0x3f}); got != 1.0 { // 1.0f LE
		t.Errorf("F32 1.0: got %v", got)
	}
	if got := f32([]byte{0x00, 0x00, 0x80, 0xbf}); got != -1.0 {
		t.Errorf("F32 -1.0: got %v", got)
	}
	f64 := Lookup("F64").Decode
	if got := f64([]byte{0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40}); math.Abs(got-math.Pi) > 1e-15 {
		t.Errorf("F64 pi: got %v", got)
	}
}

func TestDecodeF16(t *testing.T) {
	dec := Lookup("F16").Decode
	cases := []struct {
		bits uint16
		want float64
	}{
		{0x3c00, 1.0},
		{0xbc00, -1.0},
		{0x3555, 0.333251953125},     // closest F16 to 1/3, exact in float64
		{0x0001, math.Ldexp(1, -24)}, // smallest subnormal
		{0x0000, 0.0},
		{0x8000, math.Copysign(0, -1)}, // negative zero
		{0x7bff, 65504},                // largest finite half
	}
	for _, c := range cases {
		got := dec([]byte{byte(c.bits), byte(c.bits >> 8)})
		if got != c.want {
			t.Errorf("F16 %#04x: got %v, want %v", c.bits, got, c.want)
		}
	}
	// Specials cannot go through the == table: NaN never equals itself.
	if got := dec([]byte{0x00, 0x7c}); !math.IsInf(got, 1) {
		t.Errorf("F16 +Inf: got %v", got)
	}
	if got := dec([]byte{0x00, 0xfc}); !math.IsInf(got, -1) {
		t.Errorf("F16 -Inf: got %v", got)
	}
	if got := dec([]byte{0x01, 0x7c}); !math.IsNaN(got) {
		t.Errorf("F16 NaN: got %v", got)
	}
}

func TestDecodeBF16(t *testing.T) {
	dec := Lookup("BF16").Decode
	cases := []struct {
		bits uint16
		want float64
	}{
		{0x3f80, 1.0},
		{0xbf80, -1.0},
		{0x4049, float64(math.Float32frombits(0x40490000))}, // ~pi truncated
		{0x0000, 0.0},
	}
	for _, c := range cases {
		got := dec([]byte{byte(c.bits), byte(c.bits >> 8)})
		if got != c.want {
			t.Errorf("BF16 %#04x: got %v, want %v", c.bits, got, c.want)
		}
	}
	if got := dec([]byte{0x80, 0x7f}); !math.IsInf(got, 1) {
		t.Errorf("BF16 +Inf: got %v", got)
	}
	if got := dec([]byte{0xc1, 0x7f}); !math.IsNaN(got) {
		t.Errorf("BF16 NaN: got %v", got)
	}
}

func TestDecodeSignedIntegers(t *testing.T) {
	if got := Lookup("I8").Decode([]byte{0x80}); got != -128 {
		t.Errorf("I8 min: got %v", got)
	}
	if got := Lookup("I16").Decode([]byte{0xff, 0xff}); got != -1 {
		t.Errorf("I16 -1: got %v", got)
	}
	if got := Lookup("I32").Decode([]byte{0xfe, 0xff, 0xff, 0xff}); got != -2 {
		t.Errorf("I32 -2: got %v", got)
	}
	if got := Lookup("I64").Decode([]byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}); got != -1 {
		t.Errorf("I64 -1: got %v", got)
	}
}

func TestDecodeUnsignedAndBool(t *testing.T) {
	if got := Lookup("U8").Decode([]byte{0xff}); got != 255 {
		t.Errorf("U8 255: got %v", got)
	}
	if got := Lookup("U16").Decode([]byte{0x34, 0x12}); got != 0x1234 {
		t.Errorf("U16: got %v", got)
	}
	if got := Lookup("U32").Decode([]byte{0x00, 0x00, 0x00, 0x80}); got != float64(1)*math.Pow(2, 31) {
		t.Errorf("U32 2^31: got %v", got)
	}
	if got := Lookup("BOOL").Decode([]byte{0x01}); got != 1 {
		t.Errorf("BOOL true: got %v", got)
	}
	if got := Lookup("BOOL").Decode([]byte{0x00}); got != 0 {
		t.Errorf("BOOL false: got %v", got)
	}
}
