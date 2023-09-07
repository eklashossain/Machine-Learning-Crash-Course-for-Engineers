# Declaring global variable
globalvariable = 63

# Building a function 
def insidescope():
    localvariable = 63
    print("Inside the scope: ")
    print("This is global variable: ", globalvariable)
    print("This is local variable: ", localvariable)
    return localvariable

# Calling the function
insidescope()

# Printing the variables
print("Outside the scope: ")
print("This is global variable: ", globalvariable)
print("This is local variable: ", localvariable)
