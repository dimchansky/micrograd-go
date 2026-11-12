package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"

	mg "example.com/micrograd"
	"github.com/fogleman/gg"
	"golang.org/x/image/font"
	"golang.org/x/image/font/gofont/gobold"
	"golang.org/x/image/font/gofont/goregular"
	"golang.org/x/image/font/opentype"
)

type bounds struct{ xmin, xmax, ymin, ymax float64 }
type viewport struct{ x, y, w, h float64 }

func plotBounds(data []sample, aspect float64) bounds {
	b := bounds{data[0].x, data[0].x, data[0].y, data[0].y}
	for _, p := range data {
		b.xmin, b.xmax = math.Min(b.xmin, p.x), math.Max(b.xmax, p.x)
		b.ymin, b.ymax = math.Min(b.ymin, p.y), math.Max(b.ymax, p.y)
	}
	b.xmin, b.xmax, b.ymin, b.ymax = b.xmin-0.55, b.xmax+0.55, b.ymin-0.55, b.ymax+0.55
	// Keep equal units on the two axes, so the moons are not distorted.
	w, h := b.xmax-b.xmin, b.ymax-b.ymin
	if w/h < aspect {
		extra := (h*aspect - w) / 2
		b.xmin, b.xmax = b.xmin-extra, b.xmax+extra
	} else {
		extra := (w/aspect - h) / 2
		b.ymin, b.ymax = b.ymin-extra, b.ymax+extra
	}
	return b
}

func (v viewport) point(b bounds, x, y float64) (float64, float64) {
	return v.x + (x-b.xmin)/(b.xmax-b.xmin)*v.w,
		v.y + (b.ymax-y)/(b.ymax-b.ymin)*v.h
}

type field struct {
	cols, rows int
	scores     []float64
}

func sampleField(model *mg.MLP, b bounds, cols, rows int) (field, error) {
	f := field{cols, rows, make([]float64, (cols+1)*(rows+1))}
	for row := 0; row <= rows; row++ {
		y := b.ymax - float64(row)/float64(rows)*(b.ymax-b.ymin)
		for col := 0; col <= cols; col++ {
			x := b.xmin + float64(col)/float64(cols)*(b.xmax-b.xmin)
			score := forward(model, x, y)
			if math.IsNaN(score.Data) || math.IsInf(score.Data, 0) {
				return field{}, fmt.Errorf("non-finite score on decision grid")
			}
			f.scores[row*(cols+1)+col] = score.Data
		}
	}
	return f, nil
}

func (f field) at(col, row int) float64 { return f.scores[row*(f.cols+1)+col] }

// interpolate uses two linear triangles per cell. The fill and contour use
// exactly the same approximation of the sampled model scores.
func (f field) interpolate(x, y float64) float64 {
	col, row := min(int(x), f.cols-1), min(int(y), f.rows-1)
	dx, dy := x-float64(col), y-float64(row)
	a, b, c, d := f.at(col, row), f.at(col+1, row), f.at(col+1, row+1), f.at(col, row+1)
	if dx >= dy {
		return a*(1-dx) + b*(dx-dy) + c*dy
	}
	return a*(1-dy) + c*dx + d*(dy-dx)
}

type vertex struct{ x, y, score float64 }

// zeroSegment intersects the score=0 contour with one triangle.
func zeroSegment(t [3]vertex) ([2]vertex, bool) {
	var line [2]vertex
	count := 0
	for i, a := range t {
		b := t[(i+1)%3]
		if (a.score > 0) == (b.score > 0) {
			continue
		}
		weight := a.score / (a.score - b.score)
		line[count] = vertex{x: a.x + weight*(b.x-a.x), y: a.y + weight*(b.y-a.y)}
		count++
	}
	return line, count == 2
}

func drawField(dc *gg.Context, v viewport, f field) {
	bg := image.NewRGBA(image.Rect(0, 0, int(v.w), int(v.h)))
	negative := color.RGBA{227, 239, 250, 255}
	positive := color.RGBA{255, 237, 219, 255}
	for y := 0; y < bg.Bounds().Dy(); y++ {
		for x := 0; x < bg.Bounds().Dx(); x++ {
			score := f.interpolate((float64(x)+0.5)/v.w*float64(f.cols),
				(float64(y)+0.5)/v.h*float64(f.rows))
			c := negative
			if score > 0 {
				c = positive
			}
			bg.SetRGBA(x, y, c)
		}
	}
	dc.DrawImage(bg, int(v.x), int(v.y))

	dc.SetHexColor("#233549")
	dc.SetLineWidth(2)
	for row := 0; row < f.rows; row++ {
		for col := 0; col < f.cols; col++ {
			x, y := v.x+float64(col)*v.w/float64(f.cols), v.y+float64(row)*v.h/float64(f.rows)
			w, h := v.w/float64(f.cols), v.h/float64(f.rows)
			a := vertex{x, y, f.at(col, row)}
			b := vertex{x + w, y, f.at(col+1, row)}
			c := vertex{x + w, y + h, f.at(col+1, row+1)}
			d := vertex{x, y + h, f.at(col, row+1)}
			for _, triangle := range [][3]vertex{{a, b, c}, {a, c, d}} {
				if line, ok := zeroSegment(triangle); ok {
					dc.DrawLine(line[0].x, line[0].y, line[1].x, line[1].y)
					dc.Stroke()
				}
			}
		}
	}
}

type faceKey struct {
	size float64
	bold bool
}

type painter struct {
	dc    *gg.Context
	faces map[faceKey]font.Face
}

func newPainter() (*painter, error) {
	p := &painter{dc: gg.NewContext(1440, 1000), faces: make(map[faceKey]font.Face)}
	for bold, data := range map[bool][]byte{false: goregular.TTF, true: gobold.TTF} {
		f, err := opentype.Parse(data)
		if err != nil {
			p.close()
			return nil, err
		}
		for _, size := range []float64{14, 16, 18, 20, 24, 28, 40} {
			face, err := opentype.NewFace(f, &opentype.FaceOptions{Size: size, DPI: 72, Hinting: font.HintingNone})
			if err != nil {
				p.close()
				return nil, err
			}
			p.faces[faceKey{size, bold}] = face
		}
	}
	return p, nil
}

func (p *painter) close() {
	for _, face := range p.faces {
		_ = face.Close()
	}
}

func (p *painter) text(s string, x, y, size float64, bold bool, ink string, align float64) {
	p.dc.SetFontFace(p.faces[faceKey{size, bold}])
	p.dc.SetHexColor(ink)
	w, _ := p.dc.MeasureString(s)
	p.dc.DrawString(s, x-align*w, y)
}

func marker(dc *gg.Context, x, y, label float64) {
	dc.SetHexColor("#286bb5")
	dc.DrawCircle(x, y, 5.5)
	if label > 0 {
		dc.ClearPath()
		dc.SetHexColor("#cb6228")
		dc.DrawRegularPolygon(4, x, y, 7, 0)
	}
	dc.FillPreserve()
	dc.SetHexColor("#ffffff")
	dc.SetLineWidth(1.3)
	dc.Stroke()
}

func drawAxes(p *painter, v viewport, b bounds) {
	dc := p.dc
	dc.SetLineWidth(1)
	for x := math.Ceil(b.xmin*2) / 2; x <= b.xmax; x += 0.5 {
		px, _ := v.point(b, x, 0)
		dc.SetRGBA(0.25, 0.35, 0.45, 0.12)
		dc.DrawLine(px, v.y, px, v.y+v.h)
		dc.Stroke()
		p.text(fmt.Sprintf("%.1f", x), px, v.y+v.h+26, 14, false, "#64748b", 0.5)
	}
	for y := math.Ceil(b.ymin*2) / 2; y <= b.ymax; y += 0.5 {
		_, py := v.point(b, 0, y)
		dc.SetRGBA(0.25, 0.35, 0.45, 0.12)
		dc.DrawLine(v.x, py, v.x+v.w, py)
		dc.Stroke()
		p.text(fmt.Sprintf("%.1f", y), v.x-16, py+5, 14, false, "#64748b", 1)
	}
	dc.SetHexColor("#bcc8d3")
	dc.DrawRectangle(v.x, v.y, v.w, v.h)
	dc.Stroke()
	p.text("x", v.x+v.w/2, v.y+v.h+54, 18, false, "#475569", 0.5)
	p.text("y", v.x-45, v.y-14, 18, false, "#475569", 0.5)
}

func drawLoss(p *painter, v viewport, history []metrics) {
	dc := p.dc
	low, high := math.Inf(1), math.Inf(-1)
	for _, m := range history {
		value := math.Log10(math.Max(m.total(), 1e-12))
		low, high = math.Min(low, value), math.Max(high, value)
	}
	low, high = math.Floor(low), math.Ceil(high)
	if low == high {
		high++
	}
	ypos := func(value float64) float64 {
		return v.y + (high-value)/(high-low)*v.h
	}
	for e := low; e <= high; e++ {
		y := ypos(e)
		dc.SetHexColor("#e3e9ee")
		dc.SetLineWidth(1)
		dc.DrawLine(v.x, y, v.x+v.w, y)
		dc.Stroke()
		p.text(fmt.Sprintf("%g", math.Pow(10, e)), v.x-12, y+5, 14, false, "#64748b", 1)
	}
	steps := max(1, len(history)-1)
	for _, step := range []int{0, steps / 2, steps} {
		x := v.x + float64(step)/float64(steps)*v.w
		p.text(fmt.Sprint(step), x, v.y+v.h+25, 14, false, "#64748b", 0.5)
		if len(history) == 1 {
			break
		}
	}
	dc.SetHexColor("#334f7d")
	dc.SetLineWidth(2.3)
	for i, m := range history {
		x := v.x + float64(i)/float64(steps)*v.w
		y := ypos(math.Log10(math.Max(m.total(), 1e-12)))
		if i == 0 {
			dc.MoveTo(x, y)
		} else {
			dc.LineTo(x, y)
		}
	}
	dc.Stroke()
	last := history[len(history)-1]
	x := v.x + float64(len(history)-1)/float64(steps)*v.w
	dc.DrawCircle(x, ypos(math.Log10(math.Max(last.total(), 1e-12))), 4)
	dc.Fill()
	p.text("SGD updates", v.x+v.w/2, v.y+v.h+51, 14, false, "#64748b", 0.5)
}

func render(o options, model *mg.MLP, data []sample, history []metrics) error {
	p, err := newPainter()
	if err != nil {
		return err
	}
	defer p.close()
	dc := p.dc
	dc.SetHexColor("#f9fbfd")
	dc.Clear()
	p.text("Learning a nonlinear boundary", 64, 73, 40, true, "#172b42", 0)
	p.text("Two moons  /  scalar autograd in Go  /  MLP 2 → 16 → 16 → 1", 66, 112, 18, false, "#607186", 0)

	final := history[len(history)-1]
	for i, stat := range []struct{ label, value string }{
		{"TRAINING ACCURACY", fmt.Sprintf("%.1f%%", final.accuracy*100)},
		{"TRAINING POINTS", fmt.Sprint(len(data))},
		{"SGD UPDATES", fmt.Sprint(o.steps)},
		{"GAUSSIAN NOISE", fmt.Sprintf("σ = %.2f", o.noise)},
	} {
		x := 66 + float64(i)*335
		p.text(stat.label, x, 164, 14, false, "#718096", 0)
		p.text(stat.value, x, 201, 28, true, "#172b42", 0)
	}
	dc.SetHexColor("#dbe4ec")
	dc.SetLineWidth(1)
	dc.DrawLine(64, 224, 1376, 224)
	dc.Stroke()

	p.text("Decision regions", 66, 267, 24, true, "#172b42", 0)
	p.text("Total loss", 1040, 267, 24, true, "#172b42", 0)
	p.text("Mean hinge + L2  ·  logarithmic scale", 1040, 294, 14, false, "#64748b", 0)
	v := viewport{112, 310, 850, 560}
	b := plotBounds(data, v.w/v.h)
	f, err := sampleField(model, b, o.grid, max(2, int(math.Round(float64(o.grid)*v.h/v.w))))
	if err != nil {
		return err
	}
	drawField(dc, v, f)
	drawAxes(p, v, b)
	for _, point := range data {
		x, y := v.point(b, point.x, point.y)
		marker(dc, x, y, point.label)
		score := forward(model, point.x, point.y)
		if (score.Data > 0) != (point.label > 0) {
			dc.SetHexColor("#233549")
			dc.SetLineWidth(1.8)
			dc.DrawCircle(x, y, 10)
			dc.Stroke()
		}
	}
	drawLoss(p, viewport{1090, 330, 270, 215}, history)
	p.text("How to read this", 1040, 652, 20, true, "#172b42", 0)
	marker(dc, 1050, 686, -1)
	p.text("Class −1", 1072, 692, 18, false, "#475569", 0)
	marker(dc, 1220, 686, 1)
	p.text("Class +1", 1242, 692, 18, false, "#475569", 0)
	dc.SetHexColor("#233549")
	dc.SetLineWidth(2)
	dc.DrawLine(1042, 725, 1060, 725)
	dc.Stroke()
	p.text("Decision boundary: score = 0", 1072, 731, 16, false, "#475569", 0)
	p.text("Background = predicted class.", 1040, 772, 16, false, "#64748b", 0)
	p.text("Points = training labels.", 1040, 799, 16, false, "#64748b", 0)
	p.text("Ring = misclassified point.", 1040, 826, 16, false, "#64748b", 0)
	p.text("Accuracy is measured on training data.", 1040, 873, 14, false, "#64748b", 0)

	p.text(fmt.Sprintf("ReLU hidden layers · linear output · L2 %.4g · seed %d", o.l2, o.seed),
		66, 962, 16, false, "#607186", 0)
	p.text("Based on karpathy/micrograd", 1374, 962, 16, false, "#607186", 1)
	return savePNG(o.out, dc.Image())
}

func savePNG(path string, img image.Image) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, ".moons-*.png")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if err := png.Encode(f, img); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}
