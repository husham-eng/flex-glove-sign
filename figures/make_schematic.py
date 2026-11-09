import schemdraw, schemdraw.elements as elm
schemdraw.config(fontsize=11)
fingers=["Thumb","Index","Middle","Ring","Little","Idx-Mid\nabduction","Elbow"]
with schemdraw.Drawing(file='figures/fig_circuit_schematic.png', dpi=300, show=False) as d:
    N=len(fingers); dx=2.6; xr=(N-1)*dx+0.8
    d+=elm.Line().at((-0.6,6)).to((xr,6))
    d+=elm.Label().at((-0.6,6)).label('+5 V (Mega 5V pin)',loc='left',fontsize=10)
    d+=elm.Line().at((-0.6,0)).to((xr,0))
    d+=elm.Label().at((-0.6,0)).label('GND',loc='left',fontsize=10)
    d+=elm.Ground().at((3*dx,0))
    for i,f in enumerate(fingers):
        x=i*dx
        d+=elm.Dot().at((x,6))
        d+=elm.ResistorVar().at((x,6)).down().length(3).label(f'Flex {i+1}\n{f}',loc='bottom',fontsize=8)
        d+=elm.Dot(); n=d.here
        d+=elm.Resistor().down().length(3).label('47 kΩ',loc='bottom',fontsize=8)
        d+=elm.Dot()
        d+=elm.Line().at(n).right(0.5)
        d+=elm.Tag().right().label(f'A{i}',fontsize=9).scale(0.65).color('blue')
    ax0=xr+3.2
    mega=elm.Ic(pins=[elm.IcPin(name=f'A{i}',side='left') for i in range(N)]+
              [elm.IcPin(name='SDA (pin 20)',side='right',slot='3/3'),elm.IcPin(name='SCL (pin 21)',side='right',slot='2/3'),
               elm.IcPin(name='USB',side='right',slot='1/3')],
              size=(6.0,6.6),pinspacing=0.8)
    d+=mega.at((ax0,0.6))
    d+=elm.Label().at((ax0+3.2,6.7)).label('Arduino Mega 2560',fontsize=12)
    d+=elm.Label().at((ax0+3.2,6.15)).label('(10-bit ADC, Vref = 5 V)',fontsize=9)
    for i in range(N):
        d+=elm.Line().at(getattr(mega,f'A{i}')).left(0.7)
        d+=elm.Tag().left().label(f'A{i}',fontsize=9).scale(0.65).color('blue')
    d+=elm.Line().at(mega['SDA (pin 20)']).right(2.5); d+=elm.Label().label('SDA',loc='right',fontsize=10)
    d+=elm.Line().at(mega['SCL (pin 21)']).right(2.5); d+=elm.Label().label('SCL',loc='right',fontsize=10)
    sx=mega['SDA (pin 20)'][0]+2.5
    d+=elm.Label().at((sx+1.2,mega['SDA (pin 20)'][1]-0.45)).label('→ MPU-6050 IMU (optional)\n   VCC→5V, GND→GND',loc='right',fontsize=9)
    d+=elm.Line().at(mega['USB']).right(2.5); d+=elm.Label().label('USB → PC (Python: logging, features, RF/XGBoost)',loc='right',fontsize=9)
    d+=elm.Label().at((-0.6,-1.5)).label('Each channel: V_out = 5 V × 47 kΩ / (47 kΩ + R_flex).  Bending raises R_flex, so V_out falls.  Optional 100 nF from each A-pin to GND.',loc='right',fontsize=9)
