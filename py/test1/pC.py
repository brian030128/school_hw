



nums = input()
nums = [int(e) for e in nums.split(' ')]

operations = []

while True:
    line = input()
    if line == 'q':
        break
    args = line.split(' ')
    a = int(args[0])
    b = int(args[1])
    if a < 0:
        continue
    if b >= len(nums):
        continue
    operations.append((a, b))

operations.reverse()

for op in operations:
    w = []
    w.extend(nums[op[0]:op[1]+1])
    w.reverse()
    nums[op[0]:op[1]+1] = w
nums = [str(e) for e in nums]

print(' '.join(nums))