"""Draws graph-forward.svg and graph-backward.svg: the computation graph of
L = (a + b) · c with a = 2, b = 1, c = −2, before and after the backward pass.

Run from anywhere: python3 docs/images/graph.py
"""
import os
HERE=os.path.dirname(os.path.abspath(__file__))  # write the SVG next to this script
W,H=760,385
ink="#0f172a"; muted="#64748b"; grad="#c2410c"; local="#7c3aed"; edge="#94a3b8"
# name, x, y, data, grad line
G="grad"; P="local"; K="ink"
# name -> (x, y, value, backward lines); each line is a list of (text, colour key)
nodes={
 "a":(30,40,"2",[[("∂L/∂a = ∂L/∂d · ∂d/∂a",G)],[("= ",K),("−2",G),(" · ",K),("1",P),(" = ",K),("−2",G)]]),
 "b":(30,190,"1",[[("∂L/∂b = ∂L/∂d · ∂d/∂b",G)],[("= ",K),("−2",G),(" · ",K),("1",P),(" = ",K),("−2",G)]]),
 "d":(300,111,"3",[[("∂L/∂d = ",G),("1",G),(" · ",K),("(−2)",P),(" = −2",G)]]),
 "c":(300,230,"−2",[[("∂L/∂c = ",G),("1",G),(" · ",K),("3",P),(" = 3",G)]]),
 "L":(570,171,"−6",[[("∂L/∂L = 1",G)],[("start here",K)]]),
}
BW,BH=140,78
OPS0={"+":(235,150),"×":(505,210)}
ops=dict(OPS0)
def box(n,back):
    x,y,data,lines=nodes[n]
    if back and n in "ab": x-=18
    w=BW+(26 if back and n in "ab" else 0); h=BH+(20 if back else 0)
    label={"d":"d = a + b","L":"L = d · c"}.get(n,n)
    s=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#f8fafc" stroke="{ink}" stroke-width="1.5"/>'
    s+=f'<text x="{x+12}" y="{y+24}" font-weight="600">{label}</text>'
    s+=f'<text x="{x+12}" y="{y+46}">value {data}</text>'
    if back:
        col={"grad":grad,"local":local,"ink":muted}
        for i,line in enumerate(lines):
            spans="".join(f'<tspan fill="{col[c]}">{t}</tspan>' for t,c in line)
            s+=f'<text x="{x+12}" y="{y+68+i*19}" font-size="13" font-weight="600" xml:space="preserve">{spans}</text>'
    return s
def op(o):
    x,y=ops[o]; return f'<circle cx="{x}" cy="{y}" r="20" fill="#ffffff" stroke="{ink}" stroke-width="1.5"/><text x="{x}" y="{y+6}" text-anchor="middle" font-size="18">{o}</text>'
def arrow(x1,y1,x2,y2): return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{edge}" stroke-width="2" marker-end="url(#fwd)"/>'
BACK=False
def mid(n,side):
    x,y,*_=nodes[n]; h=BH+(20 if BACK else 0); w=BW+(8 if BACK and n in "ab" else 0)
    return (x+w,y+h/2) if side=="r" else (x,y+h/2)
def toward(src,dst,r=22):
    import math
    (x1,y1),(x2,y2)=src,dst; dx,dy=x2-x1,y2-y1; l=math.hypot(dx,dy)
    return x2-dx/l*r, y2-dy/l*r
def build(back):
    global BACK; BACK=back
    ops.update({k:(x,y+(10 if back else 0)) for k,(x,y) in OPS0.items()})
    p=[]
    for n,o in (("a","+"),("b","+"),("d","×"),("c","×")):
        s=mid(n,"r"); t=toward(s,ops[o]); p.append(arrow(*s,*t))
    for o,n in (("+","d"),("×","L")):
        x,y=ops[o]; tx,ty=mid(n,"l"); p.append(arrow(x+20,y,tx-3,ty))
    p+= [box(n,back) for n in nodes]+[op(o) for o in ops]
    if back:
        p.append(f'<g fill="{local}" font-size="13">'
          '<text x="196" y="104">∂d/∂a = 1</text>'
          '<text x="196" y="232">∂d/∂b = 1</text>'
          '<text x="452" y="150">∂L/∂d = c = −2</text>'
          '<text x="478" y="272">∂L/∂c = d = 3</text></g>')
        p.append(f'<line x1="700" y1="355" x2="60" y2="355" stroke="{grad}" stroke-width="2" marker-end="url(#back)"/>')
        p.append(f'<text x="380" y="348" text-anchor="middle" fill="{grad}">backward, against the arrows: start with 1 at L, multiply by the local derivatives on the edges</text>')
    else:
        p.append(f'<line x1="60" y1="362" x2="700" y2="362" stroke="{muted}" stroke-width="2" marker-end="url(#fwdbig)"/>')
        p.append(f'<text x="380" y="348" text-anchor="middle" fill="{muted}">forward: compute values from the inputs to L</text>')
    return p
def svg(back):
    title = "L = (a + b) · c: values and gradients" if back else "L = (a + b) · c: forward values"
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif" font-size="15" fill="{ink}">
  <!-- {title}. Inputs a = 2, b = 1, c = −2. -->
  <rect width="{W}" height="{H}" rx="12" fill="#ffffff"/>
  <defs>
    <marker id="fwd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{edge}"/></marker>
    <marker id="fwdbig" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{muted}"/></marker>
    <marker id="back" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{grad}"/></marker>
  </defs>
  {chr(10).join("  "+x for x in build(back)).strip()}
</svg>
'''
open(os.path.join(HERE,"graph-forward.svg"),"w").write(svg(False))
open(os.path.join(HERE,"graph-backward.svg"),"w").write(svg(True))
