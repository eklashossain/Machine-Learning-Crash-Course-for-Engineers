# creating a list/array
colors = ['red', 'black', 'orange', 'blue']

# Accessing array elements
print("The first color in the array is", colors[0])  ## red
print("The second color in the array is", colors[1]) ## black

# looping through all the elements in the array
for color in colors:
    print(color) ## red, black, orange, blue
    
# Slicing
print(colors[1:3])  ## ['black', 'orange']
print(colors[0:4])  ## ['red', 'black', 'orange', 'blue']
print("---------------")

# Updating
colors[2] = 'green'
for color in colors:
   print(color)     ## red, black, green, blue
print("---------------")
  
colors.append('pink')
for color in colors:
   print(color)     ## red, black, green, blue, pink
print("---------------")
   
colors.insert(3, 'yellow')
for color in colors:
   print(color)   ## red, black, green, yellow, blue, pink
print("---------------")

# Delete
colors.remove("pink")
for color in colors:
   print(color)     ## red, black, green, yellow, blue
print("---------------")

colors.pop(2)
for color in colors:
   print(color)     ## red, black, yellow, blue