import copy

w = [i+1 for i in range(20)]
w.reverse()

all_scores = [
    [(i) for i in w],
    [2*(i) for i in w],
    [3*(i) for i in w]
]

state = []

global best_state
best_state = []
global best_turns
best_turns = -1


def cal_preference(shot):
    if shot[0] == 3 and shot[1] == 20:
        return 100000
    if shot[0] == 0:
        return 1000
    if shot[0] == 1:
        return shot[1]
    if shot[1] == 2:
        return 100 + shot[1]
    return 200 + shot[1]

def backtrack(state, score, turns):
    global best_turns, best_state
    if score == 0:
        if best_turns < turns:
            best_turns = turns
            best_state = copy.deepcopy(state)
        if best_turns == turns:
            a = copy.deepcopy(state)
            b = copy.deepcopy(best_state)
            a.reverse()
            b.reverse()
            replace = False
            for i in range(len(a)):
                if cal_preference(a[i]) > cal_preference(b[i]):
                    best_state = copy.deepcopy(state)
                    break
        return True

    if turns == 0:
        return False
    if score < 0:
        return False
    
    if score >= 60:
        state.append((3, 20))
        score -= 60
        turns -= 1
        backtrack(state, score, turns)
        turns += 1
        score += 60
        state.pop()

    if score >= 50:
        state.append((0, 50))
        score -= 50
        turns -= 1
        backtrack(state, score, turns)
        turns += 1
        score += 50
        state.pop()

    for multiplier in [3, 2, 1]:
        scores = all_scores[multiplier-1]
        for s in scores:
            state.append((multiplier, int(s/multiplier)))
            score -= s
            turns -= 1
            backtrack(state, score, turns)
            turns += 1
            score += s
            state.pop()

turns = int(input())
score = int(input())

for s in all_scores[1]:
    backtrack([(2, int(s/2))], score - s, turns - 1)

#print(best_state)
best_state.reverse()
for s in best_state:
    if s[0] == 0:
        print("Bull")
    if s[0] == 1:
        print(f"Single {s[1]}")
    if s[0] == 2:
        print(f"Double {s[1]}")
    if s[0] == 3:
        print(f"Triple {s[1]}")

if len(best_state) == 0:
    print("Not able to win in this round.")

