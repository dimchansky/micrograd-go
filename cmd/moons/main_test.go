package main

import (
	"math"
	"math/rand/v2"
	"testing"

	mg "github.com/dimchansky/micrograd-go"
)

func closeTo(t *testing.T, got, want float64) {
	t.Helper()
	if math.IsNaN(got) || math.Abs(got-want) > 1e-9*(1+math.Abs(want)) {
		t.Fatalf("got %g, want %g", got, want)
	}
}

func TestMoonsGeometryAndReproducibility(t *testing.T) {
	a := makeMoons(7, 0, rand.New(rand.NewPCG(1337, 0)))
	b := makeMoons(7, 0, rand.New(rand.NewPCG(1337, 0)))
	counts := map[float64]int{}
	for i, p := range a {
		if p != b[i] {
			t.Fatal("same seed produced different data")
		}
		counts[p.label]++
		if p.label < 0 {
			closeTo(t, p.x*p.x+p.y*p.y, 1)
		} else {
			closeTo(t, (p.x-1)*(p.x-1)+(p.y-0.5)*(p.y-0.5), 1)
		}
	}
	if counts[-1] != 3 || counts[1] != 4 {
		t.Fatalf("unexpected class balance: %v", counts)
	}
	// With nonzero noise the seed still determines the exact data.
	c := makeMoons(7, 0.1, rand.New(rand.NewPCG(1337, 0)))
	d := makeMoons(7, 0.1, rand.New(rand.NewPCG(1337, 0)))
	different := false
	for i := range c {
		if c[i] != d[i] {
			t.Fatal("noisy data is not reproducible")
		}
		different = different || c[i] != a[i]
	}
	if !different {
		t.Fatal("noise had no effect")
	}
}

func TestHingeGradient(t *testing.T) {
	for _, tc := range []struct{ score, label, value, grad float64 }{
		{2, 1, 0, 0}, {0, 1, 1, -1}, {2, -1, 3, 1}, {1, 1, 0, 0},
	} {
		score := mg.New(tc.score)
		loss := hinge(score, tc.label)
		loss.Backward()
		closeTo(t, loss.Data, tc.value)
		closeTo(t, score.Grad, tc.grad)
	}
}

func TestObjectiveIncludesBiasRegularization(t *testing.T) {
	model := mg.NewMLP(2, []int{1}, rand.New(rand.NewPCG(1, 2)))
	for i, value := range []float64{0.2, -0.3, 0.1} {
		model.Parameters()[i].Data = value
	}
	loss, m := objective(model, []sample{{1, 2, 1}, {-1, 0.5, -1}}, 0.1)
	closeTo(t, m.dataLoss, 1.025)
	closeTo(t, m.regLoss, 0.014)
	closeTo(t, m.accuracy, 0.5)
	closeTo(t, loss.Data, 1.039)
	loss.Backward()
	for i, want := range []float64{-0.96, -0.81, 0.02} {
		closeTo(t, model.Parameters()[i].Grad, want)
	}
}

func TestTrainLearnsSeparableClasses(t *testing.T) {
	model := mg.NewMLP(2, []int{1}, rand.New(rand.NewPCG(1, 2)))
	for _, p := range model.Parameters() {
		p.Data = 0
	}
	history, err := train(model, []sample{{-1, 0, -1}, {1, 0, 1}}, 10, 0, 1e-4, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(history) != 11 {
		t.Fatalf("got %d history entries, want 11", len(history))
	}
	last := history[len(history)-1]
	if last.total() >= history[0].total() || last.accuracy != 1 {
		t.Fatalf("training did not solve the problem: %+v", last)
	}
	// Zero steps is useful for drawing an untrained model and must be valid.
	zero, err := train(model, []sample{{1, 0, 1}}, 0, 0, 1e-4, nil)
	if err != nil || len(zero) != 1 {
		t.Fatalf("zero-step training: history=%v err=%v", zero, err)
	}
}

func TestDecisionFieldUsesConsistentCoordinates(t *testing.T) {
	model := mg.NewMLP(2, []int{1}, rand.New(rand.NewPCG(1, 2)))
	for i, value := range []float64{1, 2, 0} {
		model.Parameters()[i].Data = value
	}
	b := bounds{-1, 1, -1, 1}
	f, err := sampleField(model, b, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	// Top-left is (-1,+1), bottom-right is (+1,-1).
	closeTo(t, f.at(0, 0), 1)
	closeTo(t, f.at(2, 2), -1)
	// A linear score must remain exact inside both interpolation triangles.
	for _, p := range [][2]float64{{0.2, 0.8}, {1.7, 1.1}, {1, 1}} {
		x, y := -1+p[0], 1-p[1]
		closeTo(t, f.interpolate(p[0], p[1]), x+2*y)
	}
	line, ok := zeroSegment([3]vertex{{-1, 0, -1}, {1, 0, 1}, {0, 1, 2}})
	if !ok {
		t.Fatal("missing boundary segment")
	}
	for _, p := range line {
		closeTo(t, p.x+2*p.y, 0)
	}
}

func TestOptionsRejectInvalidNumbers(t *testing.T) {
	valid := options{samples: 100, steps: 100, grid: 120, noise: 0.1, l2: 1e-4, out: "moons.png"}
	if err := valid.validate(); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*options){
		func(o *options) { o.samples = 3 },
		func(o *options) { o.steps = -1 },
		func(o *options) { o.batch = -1 },
		func(o *options) { o.grid = 1 },
		func(o *options) { o.noise = math.NaN() },
		func(o *options) { o.l2 = math.Inf(1) },
		func(o *options) { o.out = "" },
	} {
		o := valid
		change(&o)
		if err := o.validate(); err == nil {
			t.Fatalf("accepted invalid options: %+v", o)
		}
	}
}

func TestMicrobatchesMatchFullBatch(t *testing.T) {
	data := makeMoons(11, 0.1, rand.New(rand.NewPCG(123, 0)))
	newModel := func() *mg.MLP {
		m := mg.NewMLP(2, []int{4, 1}, rand.New(rand.NewPCG(123, 1)))
		return m
	}
	full, split := newModel(), newModel()
	a, err := train(full, data, 7, 0, 1e-4, nil)
	if err != nil {
		t.Fatal(err)
	}
	b, err := train(split, data, 7, 4, 1e-4, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		closeTo(t, b[i].dataLoss, a[i].dataLoss)
		closeTo(t, b[i].regLoss, a[i].regLoss)
		closeTo(t, b[i].accuracy, a[i].accuracy)
	}
	for i, p := range full.Parameters() {
		closeTo(t, split.Parameters()[i].Data, p.Data)
	}
}
