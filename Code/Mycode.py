state = {0:"S", 1:"F", 2:"F", 3:"F", 4:"F", 5:"H", 6:"F", 7:"H", 8:"F", 9:"F", 10:"F", 11:"H", 12:"H", 13:"F", 14:"F",15:"G"}
actions = {1:"up", 2:"left", 3:"right" }
N = 16 # totoal number of states 
V_pi = []
def value_iteration(s):
    pass    

def policy_evaluation(s, a):
    pass
    # return state 

def pi(s, a):
    # r = 0 ,1
    if s == state.get("G"):
        r = 1
    else:
        r = 0
        # S_prime = 0
    # S_prime = Policy (s, actcions)
    max_action = 0
    for i in range(3):
        temp = policy_evaluation(s, actions.get(i))
        max_action = max(max_action, temp)

    S_prime = max_action
    
        
    return r, S_prime
    # photo_2023-04-27_19-05-27.jpg

def calculate_V_total():
    for i in range(N):
        V_pi.append(value_iteration(i))


