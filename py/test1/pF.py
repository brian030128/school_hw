import copy
line = input()

origin =[int(e) for e in line.split(" ")]
array = [copy.deepcopy(origin)]

cmp_times = 0


def find_uncomplete_arr(arr):
    i = 0
    for element in arr:
        if len(element) > 1:
            return i
        i+= 1
    return None

while True:
    uncomplete_arr_i = find_uncomplete_arr(array)
    if uncomplete_arr_i == None:
        break
    arr = array[uncomplete_arr_i]
    array.remove(arr)
    pivot = arr[0]
    arr.remove(pivot)
    larger = []
    smaller = []
    for element in arr:
        cmp_times += 1
        if element > pivot:
            larger.append(element)
        if element < pivot:    
            smaller.append(element)
    array.append(smaller)
    array.append(larger)
origin.sort()

print(' '.join([str(e) for e in origin]))
print(f"Times: {cmp_times}")