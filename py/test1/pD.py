class Item:
    def __init__(self, name, type, size):
        self.name = name
        self.type = type
        self.size = size

class Bin:
    def __init__(self, bin_name, capacity, supported_types):
        self.bin_name = bin_name
        self.capacity = capacity
        self.supported_types = supported_types
        self.items = []
    def can_place(self, item: Item):
        if self.supported_types == '*' or item.type in self.supported_types:
            if self.capacity >= item.size:
                return True
        return False
    def place(self, item: Item):
        assert self.can_place(item)
        self.capacity -= item.size
        if item.name not in self.items:
            self.items.append(item.name)



bins = []
unassigned_items = []

def assign_all():
    placed = []
    for item in unassigned_items:
        if item in placed:
            continue
        for bin in bins:
            if bin.can_place(item):
                bin.place(item)
                placed.append(item)
                break
    for e in placed:
        unassigned_items.remove(e)


while True:
    cmd = input()
    if cmd == "END":
        break
    args = cmd.split(" ")
    if args[0] == "BIN":
        if args[3] == '*':
            bin = Bin(args[1], int(args[2]), '*')
        else:
            bin = Bin(args[1], int(args[2]), args[3].split(','))
        bins.append(bin)
        assign_all()
    if args[0] == "ITEM":
        item = Item(args[1], args[2], int(args[3]))
        unassigned_items.append(item)
        assign_all()

#output
for bin in bins:
    if len(bin.items) > 0:
        print(f"{bin.bin_name}: {','.join(bin.items)} (remaining: {bin.capacity})")
    else:
        print(f"{bin.bin_name}: (remaining: {bin.capacity})")

if len(unassigned_items) > 0:
    print(f"Unassigned: {','.join([e.name for e in unassigned_items])}")