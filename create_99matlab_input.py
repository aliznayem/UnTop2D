# Generate 99 line topology optimization like case

nelx = 60
nely = 20
volfrac = 0.3
penal = 3
rmin = 2

f = open('mbb.txt', 'w')

# Write instruction
output_f.write('% Instructions\n\n')
output_f.write('% In NODE section, the following format is maintained\n')
output_f.write('\n% Node#  X   Y\n\n')
output_f.write('% In LOAD section\n')
output_f.write('\n% Elem#  N1  N2  N3  N4  ME NU  Thickness\n\n')
output_f.write('% In LOAD section\n')
output_f.write('% DOF#  Load 2n-1(x), 2n(y)\n\n')
output_f.write('% In BC section\n')
output_f.write('\n% DOF#  Displacement 2n-1(x), 2n(y)\n\n')
output_f.write('% In FREEZE section\n')
output_f.write('\n% Element# Put element numbers of those you that want to forcefully keep during optimization.\n')
output_f.write('% 1 indexed.\n\n')

f.write('VOLFRAC '+str(volfrac)+'\n')
f.write('PENAL '+str(penal)+'\n')
f.write('RMIN '+str(rmin)+'\n')
f.write('FREEZE_WALLS FALSE\n\n')

# Node data
f.write('NODE START\n')
node_count = 1
for i in range(nelx+1):
    for j in range(nely+1):
        f.write(str(node_count)+'    '+str(i)+'    '+str(j)+'\n')
        
        node_count += 1
f.write('NODE END\n\n')

# Element data
f.write('ELEMENT START\n')
element_count = 1
for i in range(nelx):
    for j in range(nely):
        f.write(str(element_count)+'    '+
                str(i*nely+i+j+1)+'    '+
                str((i+1)*nely+i+j+2)+'    '+
                str((i+1)*nely+i+j+3)+'    '+
                str(i*nely+i+j+2)+'    '+
                '1    '+
                '0.3    '+
                '1\n')
        
        element_count += 1
f.write('ELEMENT END\n\n')

# BC template
f.write('BC START\n')
f.write('BC END\n')

# Load template
f.write('LOAD START\n')
f.write('LOAD END\n')

# Forcefully keep elements
f.write('FREEZE START\n')
f.write('FREEZE END\n')

f.close()