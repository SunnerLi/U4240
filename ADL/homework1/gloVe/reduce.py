lengthOfRevised = 1000000

with open('text8', 'r') as f:
    string = f.read()

with open('text10', 'w') as f:
    f.write(string[:lengthOfRevised])