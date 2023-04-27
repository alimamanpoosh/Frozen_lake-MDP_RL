N =10
p = 0.4
S = [*range(0, N+1)]
A = [*range(0, N+1)]

def P(S_prime, s, a):
    if s + a == S_prime  and a <= s + (N-s) and 0 < s < N:
    
        return p
    elif s - a == S_prime  and a < s + (N-s) and 0 < s < N:
        return (1-p)

# def P(s_next, s, a):
#     if s + a == s_next and a <= min(s, N-s) and 0 < s < N and a >= 1:
#         return p
#     elif s - a == s_next and a <= min(s, N-s) and 0 < s < N and a >= 1:
#         return 1 - p
#     else:
#         return 0


def R(s,a):
    if s == N:
        return 1
    else:
        return 0


def value_iteration(S,A,P,R):
    
    V = {s : 0 for s in S}  
    optimal_policy = {s:0 for s in S}
    while True:
        old_V = V.copy()
        for s in S:
            Q = {}
            for a in A:
                # Q[a] = R(S,A) +sum([p*(r + old_V[s_]) for p,s_,r in P[s][a]])
                Q[a] = R(s,a) +sum(P(s_next, s, a) * (old_V[s_next] for s_next in S))
            V[s] = max(Q.values())
            optimal_policy[s] = max(Q, key=Q.get)
        if all(old[s] == V[s] for s in S):
            break

    return V, optimal_policy



# s = [1,2,3,4,5,6,7]
a= ['up','down','left','right']
# Q = value_iteration( s, a, transition_matrix(s, a, P), reward_matrix(s, a, R) )
# print(S)
# print(A)
v , optimal_policy = value_iteration(S,A,P,R)

print(V)

print(optimal_policy)