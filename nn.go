package micrograd

import "math/rand/v2"

// Neuron computes a weighted sum, a bias, and optionally ReLU.
type Neuron struct {
	W      []*Value
	B      *Value
	Nonlin bool
}

func NewNeuron(inputs int, nonlin bool, rng *rand.Rand) *Neuron {
	if inputs < 1 || rng == nil {
		panic("micrograd: need positive input count and a random generator")
	}
	n := &Neuron{W: make([]*Value, inputs), B: New(0), Nonlin: nonlin}
	for i := range n.W {
		n.W[i] = New(2*rng.Float64() - 1)
	}
	return n
}

func (n *Neuron) Forward(x []*Value) *Value {
	if len(x) != len(n.W) {
		panic("micrograd: wrong number of neuron inputs")
	}
	out := n.B
	for i, w := range n.W {
		out = out.Add(w.Mul(x[i]))
	}
	if n.Nonlin {
		out = out.ReLU()
	}
	return out
}

func (n *Neuron) Parameters() []*Value {
	params := make([]*Value, 0, len(n.W)+1)
	params = append(params, n.W...)
	return append(params, n.B)
}

// Layer evaluates several neurons on the same inputs.
type Layer struct{ Neurons []*Neuron }

func NewLayer(inputs, outputs int, nonlin bool, rng *rand.Rand) *Layer {
	if outputs < 1 {
		panic("micrograd: need positive output count")
	}
	l := &Layer{Neurons: make([]*Neuron, outputs)}
	for i := range l.Neurons {
		l.Neurons[i] = NewNeuron(inputs, nonlin, rng)
	}
	return l
}

func (l *Layer) Forward(x []*Value) []*Value {
	out := make([]*Value, len(l.Neurons))
	for i, n := range l.Neurons {
		out[i] = n.Forward(x)
	}
	return out
}

func (l *Layer) Parameters() []*Value {
	var params []*Value
	for _, n := range l.Neurons {
		params = append(params, n.Parameters()...)
	}
	return params
}

// MLP composes layers; hidden layers use ReLU, the last layer is linear.
type MLP struct{ Layers []*Layer }

func NewMLP(inputs int, widths []int, rng *rand.Rand) *MLP {
	if len(widths) == 0 {
		panic("micrograd: need at least one layer")
	}
	m := &MLP{Layers: make([]*Layer, len(widths))}
	for i, width := range widths {
		m.Layers[i] = NewLayer(inputs, width, i < len(widths)-1, rng)
		inputs = width
	}
	return m
}

func (m *MLP) Forward(x []*Value) []*Value {
	for _, l := range m.Layers {
		x = l.Forward(x)
	}
	return x
}

func (m *MLP) Parameters() []*Value {
	var params []*Value
	for _, l := range m.Layers {
		params = append(params, l.Parameters()...)
	}
	return params
}
