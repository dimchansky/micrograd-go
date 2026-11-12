package micrograd_test

import (
	"math/rand/v2"
	"testing"

	mg "github.com/dimchansky/micrograd-go"
)

func newRNG() *rand.Rand { return rand.New(rand.NewPCG(1, 2)) }

func TestNeuron(t *testing.T) {
	for _, nonlin := range []bool{false, true} {
		n := mg.NewNeuron(2, nonlin, newRNG())
		n.W[0].Data, n.W[1].Data, n.B.Data = 0.5, -2, 0.25
		x := []*mg.Value{mg.New(2), mg.New(-1)}
		y := n.Forward(x)
		y.Backward()
		near(t, y.Data, 3.25, 0)
		near(t, x[0].Grad, 0.5, 0)
		near(t, x[1].Grad, -2, 0)
		for i, want := range []float64{2, -1, 1} {
			near(t, n.Parameters()[i].Grad, want, 0)
		}
	}
}

func TestInactiveNeuron(t *testing.T) {
	n := mg.NewNeuron(1, true, newRNG())
	n.W[0].Data = 1
	x := mg.New(-2)
	y := n.Forward([]*mg.Value{x})
	y.Backward()
	near(t, y.Data, 0, 0)
	near(t, x.Grad, 0, 0)
	for _, p := range n.Parameters() {
		near(t, p.Grad, 0, 0)
	}
}

func TestMLPInitializationAndParameters(t *testing.T) {
	a := mg.NewMLP(3, []int{4, 4, 1}, newRNG())
	b := mg.NewMLP(3, []int{4, 4, 1}, newRNG())
	ap, bp := a.Parameters(), b.Parameters()
	if len(ap) != 41 {
		t.Fatal("expected 41 parameters")
	}
	seen := map[*mg.Value]bool{}
	for i, p := range ap {
		if seen[p] {
			t.Fatal("duplicate parameter")
		}
		seen[p] = true
		near(t, p.Data, bp[i].Data, 0)
		if p.Data < -1 || p.Data >= 1 {
			t.Fatal("weight outside initialization range")
		}
	}
	for _, i := range []int{3, 7, 11, 15, 20, 25, 30, 35, 40} {
		near(t, ap[i].Data, 0, 0)
	}
	ap[0] = nil
	if a.Parameters()[0] == nil {
		t.Fatal("Parameters exposed its internal slice")
	}
}

func TestMLPHiddenReLUAndLinearOutput(t *testing.T) {
	m := mg.NewMLP(1, []int{1, 1}, newRNG())
	for i, v := range []float64{1, 0, -2, -1} {
		m.Parameters()[i].Data = v
	}
	for _, tc := range []struct{ x, want float64 }{{1, -3}, {-1, -1}} {
		out := m.Forward([]*mg.Value{mg.New(tc.x)})
		if len(out) != 1 {
			t.Fatal("wrong output count")
		}
		near(t, out[0].Data, tc.want, 0)
	}
}

func TestMLPGradientsAgainstFiniteDifferences(t *testing.T) {
	m := mg.NewMLP(2, []int{3, 2, 1}, newRNG())
	params := m.Parameters()
	for i, p := range params {
		p.Data = 0.1 + 0.02*float64(i)
	}
	x := []*mg.Value{mg.New(0.6), mg.New(-0.2)}
	m.Forward(x)[0].Pow(2).Backward()
	// Record analytic gradients before perturbing the weights.
	grads := make([]float64, len(params))
	for i, p := range params {
		grads[i] = p.Grad
	}
	evaluate := func(x0, x1 float64) float64 {
		y := m.Forward([]*mg.Value{mg.New(x0), mg.New(x1)})[0].Data
		return y * y
	}
	const h = 1e-6
	for i, p := range params {
		original := p.Data
		p.Data = original + h
		plus := evaluate(0.6, -0.2)
		p.Data = original - h
		minus := evaluate(0.6, -0.2)
		p.Data = original
		near(t, grads[i], (plus-minus)/(2*h), 1e-6)
	}
	near(t, x[0].Grad, (evaluate(0.6+h, -0.2)-evaluate(0.6-h, -0.2))/(2*h), 1e-6)
	near(t, x[1].Grad, (evaluate(0.6, -0.2+h)-evaluate(0.6, -0.2-h))/(2*h), 1e-6)
}

func TestMLPLearnsXOR(t *testing.T) {
	m := mg.NewMLP(2, []int{8, 1}, newRNG())
	params := m.Parameters()
	xs := [][2]float64{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}}
	ys := []float64{-1, 1, 1, -1}
	for epoch := 0; epoch < 500; epoch++ {
		loss := mg.New(0)
		for i, x := range xs {
			y := m.Forward([]*mg.Value{mg.New(x[0]), mg.New(x[1])})[0]
			loss = loss.Add(y.Sub(mg.New(ys[i])).Pow(2))
		}
		loss = loss.Div(mg.New(4))
		for _, p := range params {
			p.Grad = 0
		}
		loss.Backward()
		for _, p := range params {
			p.Data -= 0.05 * p.Grad
		}
	}
	for i, x := range xs {
		y := m.Forward([]*mg.Value{mg.New(x[0]), mg.New(x[1])})[0]
		near(t, y.Data, ys[i], 1e-3)
	}
	for i, p := range m.Parameters() {
		if p != params[i] {
			t.Fatal("updates must reuse weight leaves")
		}
	}
}

func TestShapeMistakesPanic(t *testing.T) {
	for _, f := range []func(){
		func() { mg.NewMLP(2, nil, newRNG()) },
		func() { mg.NewMLP(0, []int{1}, newRNG()) },
		func() { mg.NewMLP(2, []int{0}, newRNG()) },
		func() { mg.NewNeuron(1, true, nil) },
		func() { mg.NewNeuron(2, true, newRNG()).Forward([]*mg.Value{mg.New(1)}) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Error("expected a shape error")
				}
			}()
			f()
		}()
	}
}
