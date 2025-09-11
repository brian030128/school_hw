from typing import Dict, List

class Application:
    def __init__(self, user_name, date: str):
        self.user_name = user_name
        self.date = date



class User:
    def __init__(self, type: str, name: str):
        self.name = name
        self.borrowed_books = []
        self.type = type

class State:
    def __init__(self, books: Dict[str, int]):
        self.users: Dict[str, User] = {}
        self.books: Dict[str, int] = books

        # key=lab_name, value=List[Application]
        self.lab_applications: Dict[str, List[Application]] = {}
        self.conf_applications: Dict[str, List[Application]] = {}
    
    def create_user(self, name: str, type: str):
        if name in self.users:
            raise ValueError("User already exists")
        self.users[name] = User(type, name)
    
    def search(self, book_name: str):
        if book_name in self.books:
            book_count = self.books[book_name]
            if book_count > 0:
                print(f"{book_name} has {book_count} copies left.")
            else:
                print(f"No copies of {book_name} left.")
        else:
            print(f"{book_name} is not available in the library.")
    
    def borrow(self, user_name: str, book_name: str):
        if user_name not in self.users:
            raise ValueError("User does not exist")
        if book_name not in self.books:
            print(f"{book_name} is not available in the library.")
            return
        if self.books[book_name] == 0:
            print(f"No copies of {book_name} left.")
            return
        user = self.users[user_name]
        borrow_limit = 10 if user.type == "P" else 8 if user.type == "G" else 5
        if len(user.borrowed_books) >= borrow_limit:
            print(f"{user_name} has reached the borrow limit of {borrow_limit} books.")
            return
        
        user.borrowed_books.append(book_name)
        self.books[book_name] -= 1

    def return_book(self, user_name: str, book_name: str):
        if user_name not in self.users:
            raise ValueError("User does not exist")
        if book_name not in self.books:
            print(f"{book_name} is not available in the library.")
            return
        user = self.users[user_name]
        if book_name not in user.borrowed_books:
            print(f"{user_name} did not borrow {book_name}.")
            return
        
        user.borrowed_books.remove(book_name)
        self.books[book_name] += 1

    def apply_lab(self, user_name: str, lab_name: str, date: str):
        #user_name is not allowed to reserve labs.
        if user_name not in self.users:
            raise ValueError("User does not exist")
        user = self.users[user_name]
        if user.type == "U":
            print(f"{user_name} is not allowed to reserve labs.")
            return
        #Lab lab_id is already booked on yyyy-mm-dd.
        self.lab_applications.setdefault(lab_name, [])
        for app in self.lab_applications[lab_name]:
            if app.date == date:
                print(f"Lab {lab_name} is already booked on {date}.")
                return
        self.lab_applications[lab_name].append(Application(user_name, date))
        #print(f"{user_name} has successfully reserved {lab_name} on {date}.")
    
    def apply_conf(self, user_name: str, conf_name: str, date: str):
        #user_name is not allowed to reserve conference rooms.
        #Conference Room room_id is already booked on yyyy-mm-dd.
        if user_name not in self.users:
            raise ValueError("User does not exist")
        user = self.users[user_name]
        if user.type != "P":
            print(f"{user_name} is not allowed to reserve conference rooms.")
            return
        self.conf_applications.setdefault(conf_name, [])
        for app in self.conf_applications[conf_name]:
            if app.date == date:
                print(f"Conference Room {conf_name} is already booked on {date}.")
                return
        self.conf_applications[conf_name].append(Application(user_name, date))
        #print(f"{user_name} has successfully reserved {conf_name} on {date}.")
    def show(self, user_name: str):
        if user_name not in self.users:
            raise ValueError("User does not exist")
        #user = self.users[user_name]
        #\n
        #--- user_name's Status ---
        #Borrowed Books: [book_name;, ;book_name; ...]
        #Reserved Labs: {;yyyy-mm-dd;: ;lab_id; ...}
        #Reserved Conferences: {;yyyy-mm-dd;: ;room_id; ...}
        #\n
        #For the SHOW command, print a blank line before and after the user's status information.
        #Borrowed book list is sorted alphabetically. Reserved Labs and Conferences are sorted by dates.
        user = self.users[user_name]
        borrowed_books = sorted(user.borrowed_books)
        reserved_labs = {}
        for lab_name, applications in self.lab_applications.items():
            for app in applications:
                if app.user_name == user_name:
                    reserved_labs[app.date] = lab_name
        reserved_conferences = {}
        for conf_name, applications in self.conf_applications.items():
            for app in applications:
                if app.user_name == user_name:
                    reserved_conferences[app.date] = conf_name

        

        print(f"\n--- {user_name}'s Status ---")
        print(f"Borrowed Books: {borrowed_books}")
        
        if user.type != "U":
            print(f"Reserved Labs: {reserved_labs}")
        if user.type == "P":
            print(f"Reserved Conferences: {reserved_conferences}")
        print()



#PythonBasics 2 DataStructures 5 MachineLearning 3
#CREATE_USER U Bob
#CREATE_USER G David
#CREATE_USER P Alice
#SEARCH PythonBasics
#BORROW Bob PythonBasics
#SHOW Bob
#APPLY_LAB David 2025-05-10 A
#SHOW David
#APPLY_CONFERENCE Alice 2025-05-10 B
#SHOW Alice
#EXIT

def main():
    books_input = input().strip()
    books = {}
    
    split = books_input.split()
    for i in range(0, len(split), 2):
        book_name = split[i]
        book_count = int(split[i + 1])
        books[book_name] = book_count

    state = State(books)
    while True:
        command = input().strip()
        if command == "EXIT":
            break
        parts = command.split()
        action = parts[0]
        
        if action == "CREATE_USER":
            user_type = parts[1]
            user_name = parts[2]
            state.create_user(user_name, user_type)
        
        elif action == "SEARCH":
            book_name = parts[1]
            state.search(book_name)
        
        elif action == "BORROW":
            user_name = parts[1]
            book_name = parts[2]
            state.borrow(user_name, book_name)
        
        elif action == "RETURN":
            user_name = parts[1]
            book_name = parts[2]
            state.return_book(user_name, book_name)
        
        elif action == "APPLY_LAB":
            user_name = parts[1]
            date = parts[2]
            lab_name = parts[3]
            state.apply_lab(user_name, lab_name, date)
        
        elif action == "APPLY_CONFERENCE":
            user_name = parts[1]
            date = parts[2]
            conf_name = parts[3]
            state.apply_conf(user_name, conf_name, date)
        
        elif action == "SHOW":
            user_name = parts[1]
            state.show(user_name)
        else:
            print("Invalid command")


main()