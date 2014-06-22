import radiation_pattern
name = 'bocina.txt'
pattern = radiation_pattern.radiation_pattern()
pattern.load_file(name)
pattern.plot3d()
pattern.plot_uv()	
