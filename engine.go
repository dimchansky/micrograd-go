// Package micrograd is a tiny scalar automatic-differentiation engine with a
// minimal neural-network library on top of it.
//
// Every arithmetic operation on a *Value creates a new node that remembers its
// operands. The nodes form a directed acyclic graph (the computation graph).
// Calling Backward on the final node walks that graph in reverse and applies
// the chain rule, filling Grad of every node with the derivative of the final
// node with respect to it. That is all reverse-mode autodiff (backpropagation)
// is; the rest of the package is ordinary arithmetic.
package micrograd

import "math"

// Value is one scalar node in a computation graph.
//
// Leaves are created with New: inputs, targets, constants and trainable
// weights. Every other node is the result of an operation on earlier nodes.
//
// A Value is used through a pointer. Copying the struct would create a second
// node that the graph does not know about, and its Grad would never be filled.
type Value struct {
	// Data is the number computed in the forward pass.
	Data float64

	// Grad is d(root)/d(this node), where root is the node Backward was
	// called on. Backward adds to it instead of overwriting it, because a node
	// that feeds the root through several paths receives one contribution
	// per path.
	Grad float64

	// prev holds the operands this node was computed from. Backward needs it
	// only to discover the graph; the derivative math lives in backward.
	prev []*Value

	// backward applies the local chain rule: given this node's Grad, it adds
	// the corresponding contribution to the Grad of each operand. It is nil
	// for leaves, which have no operands.
	backward func()
}

// New returns a leaf node holding data. Its Grad starts at zero.
func New(data float64) *Value { return &Value{Data: data} }

// Add returns v + other.
//
// Local derivatives: ∂(a+b)/∂a = 1 and ∂(a+b)/∂b = 1, so the gradient of the
// sum is passed to both operands unchanged.
func (v *Value) Add(other *Value) *Value {
	out := &Value{Data: v.Data + other.Data, prev: []*Value{v, other}}
	out.backward = func() {
		// Chain rule, with L the node Backward was called on:
		//   dL/dv = dL/dout · dout/dv = out.Grad · 1
		// and the same for other. Addition passes the gradient through
		// unchanged: nudging either operand by ε nudges the sum by ε.
		// += because v and other may also reach L through other nodes,
		// and the contributions of all paths are summed.
		v.Grad += out.Grad
		other.Grad += out.Grad
	}
	return out
}

// Mul returns v * other.
//
// Local derivatives: ∂(a·b)/∂a = b and ∂(a·b)/∂b = a, so each operand
// receives the gradient of the product scaled by the other operand's value.
//
// For x.Mul(x) both lines update the same x, and the two contributions add up
// to 2·x, which is exactly d(x²)/dx. This is why gradients use += .
func (v *Value) Mul(other *Value) *Value {
	out := &Value{Data: v.Data * other.Data, prev: []*Value{v, other}}
	out.backward = func() {
		// Chain rule: dL/dv = dL/dout · dout/dv, and dout/dv = other
		// because other is a constant factor when only v changes.
		// Nudging v by ε nudges v·other by other·ε. Symmetrically for other.
		v.Grad += other.Data * out.Grad
		other.Grad += v.Data * out.Grad
	}
	return out
}

// Pow returns v raised to a constant exponent.
//
// Local derivative (power rule): d(xⁿ)/dx = n·xⁿ⁻¹. The exponent is a plain
// float64, not a *Value: it is a fixed constant, so no gradient flows into it.
//
// The derivative of x⁰ is taken as 0 everywhere, including at x = 0.
func (v *Value) Pow(exponent float64) *Value {
	out := &Value{Data: math.Pow(v.Data, exponent), prev: []*Value{v}}
	out.backward = func() {
		// Chain rule with the power rule:
		//   dL/dv = dL/dout · dout/dv = out.Grad · n·vⁿ⁻¹
		// For n = 0 the output is the constant 1, whose derivative is 0,
		// so nothing is added (and 0·0⁻¹ = NaN at v = 0 is avoided).
		if exponent != 0 {
			v.Grad += exponent * math.Pow(v.Data, exponent-1) * out.Grad
		}
	}
	return out
}

// ReLU returns max(0, v), the nonlinearity used by the neural network.
//
// Local derivative: 1 where the input is positive and 0 otherwise, so the
// gradient either passes through unchanged or is blocked. At exactly 0 the
// function has no derivative; this implementation uses 0 there.
func (v *Value) ReLU() *Value {
	out := &Value{Data: math.Max(0, v.Data), prev: []*Value{v}}
	out.backward = func() {
		// Chain rule: dL/dv = out.Grad · dout/dv.
		// For v > 0, ReLU is the identity, so dout/dv = 1 and the gradient
		// passes through. For v ≤ 0 the output is the constant 0, so
		// dout/dv = 0 and nothing is added: a small change of a negative
		// input does not change the output at all.
		// out.Data > 0 holds exactly when v.Data > 0.
		if out.Data > 0 {
			v.Grad += out.Grad
		}
	}
	return out
}

// The remaining operations are built from the ones above, so they need no
// derivative code of their own: the chain rule composes the existing rules.

// Neg returns -v, computed as v · (-1).
func (v *Value) Neg() *Value { return v.Mul(New(-1)) }

// Sub returns v - other, computed as v + (-other).
func (v *Value) Sub(other *Value) *Value { return v.Add(other.Neg()) }

// Div returns v / other, computed as v · other⁻¹.
func (v *Value) Div(other *Value) *Value { return v.Mul(other.Pow(-1)) }

// Backward computes the derivative of v with respect to every node it
// depends on, and adds it to that node's Grad.
//
// A node may call its backward function only after its own Grad is final,
// that is, after every node that uses it has already passed its contribution
// down. A topological order guarantees this: it lists every node after all of
// its operands, so walking it from the end visits every consumer before the
// nodes it consumes.
//
// Contract, which the training loops in this repository follow:
//   - Build a fresh graph for every Backward call. Intermediate nodes keep the
//     Grad they received, so calling Backward twice on the same graph adds
//     the gradients twice.
//   - Set Grad of the trainable weights to zero before each update step (or
//     once before a group of mini-batches whose gradients should be summed).
//   - Change Data of the weights only after Backward. The backward closures
//     read the operands' Data, so changing it in between gives wrong
//     gradients.
func (v *Value) Backward() {
	// Depth-first search: a node is appended only after all of its operands,
	// and seen makes sure a node shared by several paths is listed once.
	var order []*Value
	seen := make(map[*Value]bool)
	var visit func(*Value)
	visit = func(n *Value) {
		if seen[n] {
			return
		}
		seen[n] = true
		for _, operand := range n.prev {
			visit(operand)
		}
		order = append(order, n)
	}
	visit(v)

	// The chain starts from dv/dv = 1 and then flows from the result towards
	// the leaves.
	v.Grad = 1
	for i := len(order) - 1; i >= 0; i-- {
		// Reverse topological order: every node that uses order[i] has
		// already run, so order[i].Grad is complete when it is passed on.
		// Leaves have no backward: there is nothing below them.
		if order[i].backward != nil {
			order[i].backward()
		}
	}
}
