import radiation_pattern
name = 'bocina.txt'
pattern = radiation_pattern.radiation_pattern()
pattern.load_file(name)
pattern.plot3d()
pattern.plot_uv()	
d = pattern.directivity()
print d
s = pattern.solid_angle()
print s
r = pattern.axial_ratio()
print r