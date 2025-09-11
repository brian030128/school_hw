



n =int( input())
for i in range(n):
    line = input()
    args = line.split(" ")
    num1 = int(args[0])
    num2 = int(args[2])

    op = args[1]

    if op == "+":
        print(num1 + num2)
    if op == "-":
        print(num1 - num2)
    if op == "*":
        print(num1*num2)
    if op == "/":
        print(num1/num2)