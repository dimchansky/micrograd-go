"""Draws triangle.svg: the tangent triangle of f(x) = x² − 4x + 5 at x = 3 for a step h = 0.4.

Both axes use the same scale, so the drawn angle has tan = slope.
Run from anywhere: python3 docs/images/triangle.py
"""
import math, os
HERE=os.path.dirname(os.path.abspath(__file__))  # write the SVG next to this script
f=lambda x:x*x-4*x+5
s=190                                   # same scale on both axes, so the drawn angle has tan = slope
X=lambda x:180+s*(x-2.55); Y=lambda y:410-s*(y-1.3)
x0,h=3.0,0.4; fx=f(x0); d=2*x0-4; rise=d*h; tan_y=fx+rise; fy=f(x0+h)
P=(X(x0),Y(fx)); R=(X(x0+h),Y(fx)); T=(X(x0+h),Y(tan_y)); Q=(X(x0+h),Y(fy))
ink="#0f172a"; muted="#64748b"; tang="#2563eb"; err="#c2410c"; tri="#7c3aed"
curve=" ".join(f"{X(x/100):.1f},{Y(f(x/100)):.1f}" for x in range(255,351))
t1,t2=2.65,3.5
a=math.atan(d); r=36
arc=f"M{P[0]+r:.1f},{P[1]:.1f} A{r},{r} 0 0 0 {P[0]+r*math.cos(a):.1f},{P[1]-r*math.sin(a):.1f}"
def fmt(v): return f"{v:g}"
base=425; left=140
svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="800" height="460" viewBox="0 0 800 460" font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif" font-size="14" fill="{ink}">
  <!-- f(x) = x² − 4x + 5 near x = 3: the tangent triangle for a step h = 0.4. Both axes use the same scale. -->
  <rect width="800" height="460" rx="12" fill="#ffffff"/>
  <!-- Guides from the points to the value labels on the left and the x labels below. -->
  <g stroke="{muted}" stroke-dasharray="4 4" stroke-width="1">
    <line x1="{left}" y1="{P[1]:.1f}" x2="{P[0]:.1f}" y2="{P[1]:.1f}"/>
    <line x1="{left}" y1="{T[1]:.1f}" x2="{T[0]:.1f}" y2="{T[1]:.1f}"/>
    <line x1="{left}" y1="{Q[1]:.1f}" x2="{Q[0]:.1f}" y2="{Q[1]:.1f}"/>
    <line x1="{P[0]:.1f}" y1="{P[1]:.1f}" x2="{P[0]:.1f}" y2="{base}"/>
    <line x1="{R[0]:.1f}" y1="{R[1]:.1f}" x2="{R[0]:.1f}" y2="{base}"/>
  </g>
  <g text-anchor="end">
    <text x="{left-8}" y="{P[1]+5:.1f}">f(x) = {fmt(fx)}</text>
    <text x="{left-8}" y="{T[1]+10:.1f}" fill="{tang}">f(x) + rise = {fmt(tan_y)}</text>
    <text x="{left-8}" y="{Q[1]-2:.1f}" fill="{err}">f(x + h) = {fmt(round(fy,4))}</text>
  </g>
  <g text-anchor="middle" fill="{muted}">
    <text x="{P[0]:.1f}" y="{base+18}">x = {fmt(x0)}</text>
    <text x="{R[0]:.1f}" y="{base+18}">x + h = {fmt(x0+h)}</text>
  </g>
  <!-- The triangle: run h, rise f′(x)·h along the tangent. -->
  <polygon points="{P[0]:.1f},{P[1]:.1f} {R[0]:.1f},{R[1]:.1f} {T[0]:.1f},{T[1]:.1f}" fill="{tri}" fill-opacity="0.10"/>
  <polyline points="{curve}" fill="none" stroke="{ink}" stroke-width="2.5"/>
  <line x1="{X(t1):.1f}" y1="{Y(fx+d*(t1-x0)):.1f}" x2="{X(t2):.1f}" y2="{Y(fx+d*(t2-x0)):.1f}" stroke="{tang}" stroke-width="2"/>
  <line x1="{P[0]:.1f}" y1="{P[1]:.1f}" x2="{R[0]:.1f}" y2="{R[1]:.1f}" stroke="{tri}" stroke-width="2.5"/>
  <line x1="{R[0]:.1f}" y1="{R[1]:.1f}" x2="{T[0]:.1f}" y2="{T[1]:.1f}" stroke="{tri}" stroke-width="2.5"/>
  <line x1="{T[0]:.1f}" y1="{T[1]:.1f}" x2="{Q[0]:.1f}" y2="{Q[1]:.1f}" stroke="{err}" stroke-width="3"/>
  <path d="{arc}" fill="none" stroke="{tri}" stroke-width="1.5"/>
  <text x="{P[0]+40:.1f}" y="{P[1]-10:.1f}" fill="{tri}" font-style="italic">α</text>
  <g fill="{tri}">
    <text x="{(P[0]+R[0])/2+12:.1f}" y="{P[1]+20:.1f}" text-anchor="middle" stroke="#ffffff" stroke-width="5" paint-order="stroke">run = h = {fmt(h)}</text>
    <text x="{R[0]+10:.1f}" y="{(R[1]+T[1])/2+5:.1f}">rise = {fmt(rise)}</text>
  </g>
  <text x="{T[0]+24:.1f}" y="{(T[1]+Q[1])/2+12:.1f}" fill="{err}">error = {fmt(round(fy-tan_y,4))}</text>
  <circle cx="{P[0]:.1f}" cy="{P[1]:.1f}" r="5" fill="{ink}"/>
  <circle cx="{T[0]:.1f}" cy="{T[1]:.1f}" r="4.5" fill="#ffffff" stroke="{tang}" stroke-width="2"/>
  <circle cx="{Q[0]:.1f}" cy="{Q[1]:.1f}" r="5" fill="{err}"/>
  <text x="{X(3.5)-4:.1f}" y="{Y(f(3.5))-8:.1f}" text-anchor="end" font-style="italic">f(x) = x² − 4x + 5</text>
  <text x="{X(t2)+6:.1f}" y="{Y(fx+d*(t2-x0))+4:.1f}" fill="{tang}">tangent</text>
  <!-- Explanation panel: one step per line, nothing skipped. -->
  <g font-size="15">
    <text x="490" y="44" font-weight="600">tan α = rise / run = f′(x)</text>
    <text x="490" y="64" fill="{muted}" font-size="13">the slope of the tangent</text>
    <text x="490" y="98" font-weight="600">run = h</text>
    <text x="490" y="118" fill="{muted}" font-size="13">we call the step to the right h</text>
    <text x="490" y="152" font-weight="600">rise = f′(x) · run = f′(x) · h</text>
    <text x="490" y="172" fill="{muted}" font-size="13">multiply both sides by the run</text>
    <text x="490" y="206" font-weight="600">rise = 2 · 0.4 = 0.8</text>
    <text x="490" y="226" fill="{muted}" font-size="13">here f′(3) = 2 and h = 0.4</text>
    <text x="490" y="260" fill="{tang}" font-weight="600">tangent: f(x) + rise = 2 + 0.8 = 2.8</text>
    <text x="490" y="294" fill="{err}" font-weight="600">curve: f(x + h) = 2.8 + error = 2.96</text>
    <text x="490" y="328" fill="{err}" font-weight="600">error = h² = 0.16</text>
    <text x="490" y="348" fill="{muted}" font-size="13">10 times smaller h: 100 times smaller error</text>
    <text x="490" y="392" font-weight="600">f(x + h) ≈ f(x) + f′(x) · h</text>
    <text x="490" y="412" fill="{muted}" font-size="13">for a small h, drop the error</text>
  </g>
</svg>
'''
open(os.path.join(HERE,"triangle.svg"),"w").write(svg)
