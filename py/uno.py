from enum import Enum

class CardType(Enum):
    NUMBER = 1
    SKIP = 2
    REVERSE = 3
    DRAW_TWO = 4
    WILD = 5
    WILD_DRAW_FOUR = 6

class Card:
      
    def __init__(self,type, color=None, number=None):
        self.type = type
        self.color = color
        self.number = number

    def __repr__(self):
        color_code = {"red": 31, "yellow": 33, "green": 32, "blue": 36}
        if self.color is None:
            ansi_code = 0
        else:
            ansi_code = color_code[self.color]
        return f"\033[{ansi_code}m{self.value}\033[0m"

class Player:
    def __init__(self):
        self.cards = []
    
    def play(self, color_bound: str, number_bound: int):
        for card in self.cards:
            if card.color == color_bound or card.number == number_bound:
                return card
        for card in self.cards:
            if card.type == CardType.SKIP and card.color == color_bound:
                return card
        
    
    def give_card(self, card):
        self.cards.append(card)



class Game:
    def __init__(self, deck):
        self.players = []
        self.turn = 0
        self.deck = deck
        self.top_card = None

    def __str__(self):
        player_str_list = [str(player) for player in self.players]
        player_str_list[self.turn] += "  <- next"

        players_str = "\n".join(player_str_list)
        game_str = f"""{"=" * 10}
    Top Card: {self.top_card}

    Players:
    {players_str}
    {"=" * 10}"""
        return game_str
    
