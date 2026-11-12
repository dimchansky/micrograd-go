package micrograd_test

import (
	"fmt"
	"math"
	"testing"

	mg "example.com/micrograd"
)

func near(t *testing.T, got, want, tolerance float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > tolerance*(1+math.Abs(want)) {
		t.Fatalf("got %.12g, want %.12g", got, want)
	}
}

func TestRepeatedOperand(t *testing.T) {
	x := mg.New(3)
	y := x.Mul(x).Add(x)
	y.Backward()
	near(t, y.Data, 12, 0)
	near(t, x.Grad, 7, 0)
}

func TestSharedIntermediate(t *testing.T) {
	x := mg.New(2)
	u := x.Mul(x)
	y := u.Mul(u).Add(u) // x^4+x^2: derivative at 2 is 36.
	y.Backward()
	near(t, y.Data, 20, 0)
	near(t, u.Grad, 9, 0)
	near(t, x.Grad, 36, 0)
}

func TestLeafRootAndZeroValue(t *testing.T) {
	var x mg.Value
	x.Backward()
	near(t, x.Data, 0, 0)
	near(t, x.Grad, 1, 0)
}

func TestReLU(t *testing.T) {
	for _, tc := range []struct{ input, value, grad float64 }{
		{-2, 0, 0}, {0, 0, 0}, {2, 2, 1},
	} {
		x := mg.New(tc.input)
		y := x.ReLU()
		y.Backward()
		near(t, y.Data, tc.value, 0)
		near(t, x.Grad, tc.grad, 0)
	}
}

func TestPowers(t *testing.T) {
	for _, tc := range []struct{ input, exponent, value, grad float64 }{
		{0, 0, 1, 0}, {4, 0.5, 2, 0.25}, {-2, 3, -8, 12},
	} {
		x := mg.New(tc.input)
		y := x.Pow(tc.exponent)
		y.Backward()
		near(t, y.Data, tc.value, 1e-12)
		near(t, x.Grad, tc.grad, 1e-12)
	}
}

func TestGradientsAgainstFiniteDifferences(t *testing.T) {
	// Independent reference: ordinary arithmetic, no autograd operations.
	f := func(x, y float64) float64 {
		return math.Pow(x*y+x, 3)/(y*y+1) + math.Max(0, x-0.7)
	}
	const h = 1e-6
	for _, xv := range []float64{-1.3, -0.2, 0.4, 1.1, 2.2} {
		for _, yv := range []float64{-1.2, -0.1, 0.8, 1.7} {
			x, y := mg.New(xv), mg.New(yv)
			out := x.Mul(y).Add(x).Pow(3).
				Div(y.Pow(2).Add(mg.New(1))).
				Add(x.Sub(mg.New(0.7)).ReLU())
			out.Backward()
			near(t, out.Data, f(xv, yv), 1e-12)
			near(t, x.Grad, (f(xv+h, yv)-f(xv-h, yv))/(2*h), 1e-6)
			near(t, y.Grad, (f(xv, yv+h)-f(xv, yv-h))/(2*h), 1e-6)
		}
	}
}

// These are the same graphs as Karpathy's test/test_engine.py, with
// independent analytic references instead of a Python/PyTorch dependency.
func TestUpstreamSanityExample(t *testing.T) {
	x := mg.New(-4)
	z := mg.New(2).Mul(x).Add(mg.New(2)).Add(x)
	q := z.ReLU().Add(z.Mul(x))
	h := z.Mul(z).ReLU()
	y := h.Add(q).Add(q.Mul(x))
	y.Backward()
	// Near x=-4, relu(z)=0 and y=3*x^3+14*x^2+14*x+4.
	v := x.Data
	near(t, y.Data, 3*v*v*v+14*v*v+14*v+4, 0)
	near(t, x.Grad, 9*v*v+28*v+14, 0)
}

func TestUpstreamMoreOpsExample(t *testing.T) {
	a, b := mg.New(-4), mg.New(2)
	c := a.Add(b)
	d := a.Mul(b).Add(b.Pow(3))
	c = c.Add(c.Add(mg.New(1)))
	c = c.Add(mg.New(1).Add(c).Add(a.Neg()))
	d = d.Add(d.Mul(mg.New(2)).Add(b.Add(a).ReLU()))
	d = d.Add(mg.New(3).Mul(d).Add(b.Sub(a).ReLU()))
	e := c.Sub(d)
	f := e.Pow(2)
	g := f.Div(mg.New(2)).Add(mg.New(10).Div(f))
	g.Backward()
	// Near (-4,2): e=4*a+3*b+3-12*a*b-12*b^3, f=e^2, g=f/2+10/f.
	av, bv := a.Data, b.Data
	ev := 4*av + 3*bv + 3 - 12*av*bv - 12*bv*bv*bv
	fv := ev * ev
	dgdf := 0.5 - 10/(fv*fv)
	near(t, g.Data, fv/2+10/fv, 1e-12)
	near(t, a.Grad, dgdf*2*ev*(4-12*bv), 1e-12)
	near(t, b.Grad, dgdf*2*ev*(3-12*av-36*bv*bv), 1e-12)
}

func TestAccumulationWithFreshGraphs(t *testing.T) {
	x := mg.New(3)
	for i := 0; i < 2; i++ {
		u := x.Mul(x) // Fresh intermediates, same parameter leaf.
		u.Add(u).Backward()
	}
	near(t, x.Grad, 24, 0)
	x.Grad = 0
	x.Sub(mg.New(1)).Pow(2).Backward()
	near(t, x.Grad, 4, 0)
	x.Data -= 0.1 * x.Grad
	x.Grad = 0
	x.Sub(mg.New(1)).Pow(2).Backward()
	near(t, x.Grad, 3.2, 1e-12)
}

func ExampleValue_Backward() {
	x := mg.New(3)
	y := x.Mul(x).Add(x)
	y.Backward()
	fmt.Println(y.Data, x.Grad)
	// Output: 12 7
}
