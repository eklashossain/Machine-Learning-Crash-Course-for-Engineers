colors = ['red', 'black', 'orange', 'blue', 'pink']
shapes = ['circle', 'square', 'triangle']

for color in colors:
   print(color)
print("---------------")   
   
for color in colors:
    if color == 'orange':
        break
    print(color)
print("---------------")

for color in colors:
    if color == 'orange':
        continue
    print(color)
print("---------------")

for color in colors:
    for shape in shapes:
        print(color, shape)
print("---------------")

for a in range(5):
  print(a)        
print("---------------")

for a in range(1,5):
  print(a)        
print("---------------")  

for a in range(1,6,2):
  print(a)        
print("---------------") 

i = 0
while i < 5:
  print(i)
  i += 1