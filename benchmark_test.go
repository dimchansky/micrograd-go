package micrograd_test

import (
	"testing"

	mg "example.com/micrograd"
)

func BenchmarkScalar(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		x := mg.New(3)
		x.Mul(x).Add(x).Backward()
	}
}

// A complete forward/backward/SGD step for 100 samples.
func BenchmarkMLPStep(b *testing.B) {
	m := mg.NewMLP(2, []int{16, 16, 1}, newRNG())
	params := m.Parameters()
	b.ReportAllocs()
	b.ResetTimer()
	for step := 0; step < b.N; step++ {
		loss := mg.New(0)
		for i := 0; i < 100; i++ {
			x := float64(i%10) / 10
			y := m.Forward([]*mg.Value{mg.New(x), mg.New(0.5 - x)})[0]
			loss = loss.Add(y.Sub(mg.New(x)).Pow(2))
		}
		loss = loss.Mul(mg.New(0.01))
		for _, p := range params {
			p.Grad = 0
		}
		loss.Backward()
		for _, p := range params {
			p.Data -= 0.0001 * p.Grad
		}
	}
}
