package preprocessing

import "unsafe"

// Tokenizer is a thin wrapper around an underlying tokenizer implementation.
// It holds an opaque handle to a tokenizer instance, which is provided and
// managed by external code and must remain valid for the lifetime of this
// memory itself.
type Tokenizer struct {
	// tokenizer points to the underlying tokenizer instance. The memory it
	// references is owned and managed by the caller or another subsystem and
	// must outlive any use of this Tokenizer. This wrapper never takes
	// ownership of the pointer and must not free it.
	// tokenizer is an opaque handle to the underlying tokenizer implementation.
	// It is stored as unsafe.Pointer because the actual object is managed outside
	// of Go (for example by an external library via cgo) and its concrete type is
	// not exposed here. This field must not be dereferenced directly in Go code.
	tokenizer unsafe.Pointer
}

// Offset represents a character offset range with [start, end] indices.
type Offset [2]uint
