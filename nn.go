package micrograd

import "math/rand/v2"

// Neuron computes ReLU(w₁x₁ + … + wₙxₙ + b), or the same sum without ReLU
// when Nonlin is false.
//
// The weights W and the bias B are ordinary leaf Values. Nothing marks them as
// trainable: they are parameters only because the training loop reads their
// Grad and updates their Data.
type Neuron struct {
	W      []*Value
	B      *Value
	Nonlin bool
}

// NewNeuron returns a neuron with the given number of inputs.
//
// Weights start uniformly random in [-1, 1). They must not all start equal:
// neurons that start identical receive identical gradients and would stay
// identical forever. The bias starts at zero; the random weights already make
// the neurons different. The random generator is passed in explicitly so that
// every run with the same seed is reproducible.
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

// Forward builds the graph of the neuron's output for inputs x.
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

// Parameters returns the neuron's weights followed by its bias.
// The slice is new, but the Values in it are the neuron's own leaves, so
// updating their Data updates the neuron.
func (n *Neuron) Parameters() []*Value {
	params := make([]*Value, 0, len(n.W)+1)
	params = append(params, n.W...)
	return append(params, n.B)
}

// Layer is a set of independent neurons that all read the same inputs.
// A layer with k neurons turns one input vector into k outputs.
type Layer struct{ Neurons []*Neuron }

// NewLayer returns a layer of outputs neurons, each with inputs inputs.
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

// Forward evaluates every neuron on x and returns their outputs in order.
// It always returns a slice, even when the layer has a single neuron.
func (l *Layer) Forward(x []*Value) []*Value {
	out := make([]*Value, len(l.Neurons))
	for i, n := range l.Neurons {
		out[i] = n.Forward(x)
	}
	return out
}

// Parameters returns the parameters of all neurons, neuron by neuron.
func (l *Layer) Parameters() []*Value {
	var params []*Value
	for _, n := range l.Neurons {
		params = append(params, n.Parameters()...)
	}
	return params
}

// MLP (multi-layer perceptron) feeds the outputs of each layer into the next.
//
// Hidden layers use ReLU. Without a nonlinearity between them, a stack of
// linear layers would collapse into one linear function and could only learn
// straight decision boundaries. The last layer is linear so the output can be
// any real number: a regression value or an unbounded classification score.
type MLP struct{ Layers []*Layer }

// NewMLP returns a network with the given number of inputs and one layer per
// entry in widths. For example, NewMLP(2, []int{16, 16, 1}, rng) maps two
// inputs through two hidden layers of 16 neurons to one output.
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

// Forward builds the graph of the network's outputs for inputs x.
func (m *MLP) Forward(x []*Value) []*Value {
	for _, l := range m.Layers {
		x = l.Forward(x)
	}
	return x
}

// Parameters returns every weight and bias, ordered layer → neuron →
// weights → bias. Training updates exactly these Values.
func (m *MLP) Parameters() []*Value {
	var params []*Value
	for _, l := range m.Layers {
		params = append(params, l.Parameters()...)
	}
	return params
}
