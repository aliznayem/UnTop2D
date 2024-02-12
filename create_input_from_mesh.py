import meshio

input_mesh_file = 'example.inp'
output_mesh_file = 'input_file.txt'

mesh = meshio.read(input_mesh_file)
output_f = open(output_mesh_file, 'w')

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

# Write config
output_f.write('VOLFRAC 0.5\nPENAL 3\nRMIN 1.2\nFREEZE_WALLS FALSE\n')

# Write nodes
output_f.write('NODE START\n')
for i_point, point in enumerate(mesh.points):
    output_f.write('{:d}    {:f}    {:f}\n'.format(i_point+1, point[0], point[1]))
output_f.write('NODE END\n')

# Write elements
output_f.write('ELEMENT START\n')
#for i_elem, elem in enumerate(mesh.cells['quad']):
for i_elem, elem in enumerate(mesh.cells[0].data):
    output_f.write('{:d}    {:d}    {:d}    {:d}    {:d}    {:f}    {:f}    {:f}\n'.format(
        i_elem+1,
        elem[0]+1,
        elem[1]+1,
        elem[2]+1,
        elem[3]+1,
        1,    # Young's modulus
        0.3,  # Poisson's ratio
        1     # thickness
    ))

output_f.write('ELEMENT END\n')

# Load template
output_f.write('LOAD START\n')
output_f.write('LOAD END\n')

# BC template
output_f.write('BC START\n')
output_f.write('BC END\n')

# Forcefully keep elements
output_f.write('FREEZE START\n')
output_f.write('FREEZE END\n')

output_f.close()
