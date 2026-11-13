"""Draws tangent.svg: f(x) = x² − 4x + 5 with its tangent lines at x = 1 and x = 3.5.

Run from anywhere: python3 docs/images/tangent.py
"""
import os
HERE=os.path.dirname(os.path.abspath(__file__))  # write the SVG next to this script
f=lambda x:x*x-4*x+5
X=lambda x:70+110*(x+0.5)          # x in [-0.5,4.5] -> 70..620
Y=lambda y:330-36*y                # f in [0,8.5] -> 330..24
pts=" ".join(f"{X(x/50):.1f},{Y(f(x/50)):.1f}" for x in range(-25,226))
def tan(x0,a,b):
    s=2*x0-4; g=lambda x:f(x0)+s*(x-x0)
    return f'<line x1="{X(a):.1f}" y1="{Y(g(a)):.1f}" x2="{X(b):.1f}" y2="{Y(g(b)):.1f}"'
ax="#64748b"; neg="#2563eb"; pos="#c2410c"
ticks="".join(f'<line x1="{X(t)}" y1="330" x2="{X(t)}" y2="335" stroke="{ax}"/><text x="{X(t)}" y="350" text-anchor="middle">{t}</text>' for t in range(0,5))
yt="".join(f'<line x1="65" y1="{Y(t)}" x2="70" y2="{Y(t)}" stroke="{ax}"/><text x="60" y="{Y(t)+4}" text-anchor="end">{t}</text>' for t in range(0,9,2))
svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420" font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif" font-size="14" fill="#334155">
  <!-- f(x) = x² − 4x + 5 with its tangent lines at x = 1 and x = 3.5. -->
  <rect width="720" height="420" rx="12" fill="#ffffff"/>
  <defs>
    <marker id="a-neg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{neg}"/></marker>
    <marker id="a-pos" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{pos}"/></marker>
  </defs>
  <line x1="70" y1="330" x2="625" y2="330" stroke="{ax}"/>
  <line x1="70" y1="335" x2="70" y2="20" stroke="{ax}"/>
  {ticks}{yt}
  <text x="640" y="334">x</text><text x="70" y="14" text-anchor="middle">f(x)</text>
  <polyline points="{pts}" fill="none" stroke="#0f172a" stroke-width="2.5"/>
  <text x="345" y="44" text-anchor="middle" font-style="italic" font-size="15">f(x) = x² − 4x + 5</text>
  {tan(1,0.15,1.85)} stroke="{neg}" stroke-width="2"/>
  {tan(3.5,2.85,4.15)} stroke="{pos}" stroke-width="2"/>
  <circle cx="{X(1)}" cy="{Y(f(1))}" r="5.5" fill="{neg}"/>
  <circle cx="{X(3.5)}" cy="{Y(f(3.5))}" r="5.5" fill="{pos}"/>
  <!-- Dashed drop lines from the points to the x axis. -->
  <line x1="{X(1)}" y1="{Y(f(1))}" x2="{X(1)}" y2="330" stroke="{neg}" stroke-dasharray="4 4"/>
  <line x1="{X(3.5)}" y1="{Y(f(3.5))}" x2="{X(3.5)}" y2="330" stroke="{pos}" stroke-dasharray="4 4"/>
  <!-- Arrows: the direction of x in which f grows. -->
  <line x1="{X(1)-6}" y1="312" x2="{X(1)-70}" y2="312" stroke="{neg}" stroke-width="2.5" marker-end="url(#a-neg)"/>
  <line x1="{X(3.5)+6}" y1="312" x2="{X(3.5)+70}" y2="312" stroke="{pos}" stroke-width="2.5" marker-end="url(#a-pos)"/>
  <g text-anchor="middle">
    <text x="345" y="84" fill="{neg}" font-weight="600">at x = 1: slope f′(1) = −2</text>
    <text x="345" y="102" fill="{neg}">tangent goes down, so f grows to the left</text>
    <text x="345" y="134" fill="{pos}" font-weight="600">at x = 3.5: slope f′(3.5) = +3</text>
    <text x="345" y="152" fill="{pos}">tangent goes up, so f grows to the right</text>
  </g>
  <text x="360" y="395" text-anchor="middle" fill="#475569">Arrows: the direction to move x to increase f. It always points the same way as the sign of the slope.</text>
</svg>
'''
open(os.path.join(HERE,"tangent.svg"),"w").write(svg)
