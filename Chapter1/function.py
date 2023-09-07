#defining function
def calculate_mean1(list):
  a = len(list)
  sum = 0
  for i in list:
    sum += i
  result1 = sum / a
  return result1 

def calculate_mean2(list):
  a = len(list)
  sum = 0
  for i in list:
    sum += i
  result2 = sum / a
  print("The mean calculated from the second function: ", result2)

list_of_numbers = [1, 2, 3, 4, 5, 6]
result = calculate_mean1(list_of_numbers)
print("The mean calculated from the first function: ", result)
calculate_mean2(list_of_numbers)
