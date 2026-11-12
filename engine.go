// Package micrograd is a small, educational scalar autograd engine.
package micrograd

import "math"

// Value is one node in a computation graph. Leaves include inputs and weights.
// Change weight Data only after backward; do not copy Values or use them concurrently.
type Value struct {
	Data     float64
	Grad     float64
	prev     []*Value
	backward func()
}

func New(data float64) *Value { return &Value{Data: data} }

func (v *Value) Add(other *Value) *Value {
	out := &Value{Data: v.Data + other.Data, prev: []*Value{v, other}}
	out.backward = func() {
		v.Grad += out.Grad
		other.Grad += out.Grad
	}
	return out
}

func (v *Value) Mul(other *Value) *Value {
	out := &Value{Data: v.Data * other.Data, prev: []*Value{v, other}}
	out.backward = func() {
		v.Grad += other.Data * out.Grad
		other.Grad += v.Data * out.Grad
	}
	return out
}

func (v *Value) Pow(exponent float64) *Value {
	out := &Value{Data: math.Pow(v.Data, exponent), prev: []*Value{v}}
	out.backward = func() {
		if exponent != 0 { // The derivative of x^0 is zero, even at x=0.
			v.Grad += exponent * math.Pow(v.Data, exponent-1) * out.Grad
		}
	}
	return out
}

func (v *Value) ReLU() *Value {
	out := &Value{Data: math.Max(0, v.Data), prev: []*Value{v}}
	out.backward = func() {
		if out.Data > 0 {
			v.Grad += out.Grad
		}
	}
	return out
}

func (v *Value) Neg() *Value             { return v.Mul(New(-1)) }
func (v *Value) Sub(other *Value) *Value { return v.Add(other.Neg()) }
func (v *Value) Div(other *Value) *Value { return v.Mul(other.Pow(-1)) }

// Backward adds derivatives to Grad, as in Karpathy's micrograd.
// Build a fresh graph for each call. Zero weight gradients before a new update
// (or once before several microbatches), then update Data after all backward calls.
func (v *Value) Backward() {
	var order []*Value
	seen := make(map[*Value]bool)
	var visit func(*Value)
	visit = func(n *Value) {
		if seen[n] {
			return
		}
		seen[n] = true
		for _, parent := range n.prev {
			visit(parent)
		}
		order = append(order, n)
	}
	visit(v)

	v.Grad = 1 // dv/dv = 1.
	for i := len(order) - 1; i >= 0; i-- {
		if order[i].backward != nil {
			order[i].backward()
		}
	}
}
