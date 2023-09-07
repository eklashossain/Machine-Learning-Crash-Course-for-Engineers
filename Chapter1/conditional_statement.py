a = 40
b = 13

if a % 2 == 0:
    print("a is even")
    
if b % 2 == 0:
    print("b is even")
else:
    print("b is odd")    
    
if a > b:
    print("a is greater")
elif b > a:
    print("b is greater")  
else:
    print("a and b are equal")
    
if a > b:
    if a % 2 == 0:
        print("a is greater and even")
    else:
        print("a is greater and odd")
elif b > a:
    print("b is greater")  
else:
    print("a and b are equal")    