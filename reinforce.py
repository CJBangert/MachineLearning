
import random
ninety = []

fifty =  [1,0,1,0,1,1,1,0,0,0]

pop = [1,2]

actions = [1,2]

expVal = 0.0000

MAX_STEPS = 100000000.0000

def getNextState(curState, action):

	if(curState == 1):

		if action == 1:

			return random.choices(pop, [.9,.1])[0]

		elif(action == 2):

			return random.choices(pop, [.5,.5])[0]
	else:

		if(action == 1):

			return random.choices(pop, [.9,.1])[0]

		else:
			return random.choices(pop, [.5,.5])[0]


def chooseAction():
	ret = random.choices(actions, [.5,.5])[0]
	return ret

	
def traverse(start, action, traversalsLeft):

	counter = 0.0

	while (traversalsLeft > 0):

		traversalsLeft = traversalsLeft - 1

		if(start == 2 and action ==2):

			act = chooseAction()
			counter = counter + 1.0

			start = getNextState(start, act)

		else:
			act = chooseAction()
			start = getNextState(start,act)
	return counter

val = traverse(1,chooseAction(), MAX_STEPS)
print(val)
exp = val / MAX_STEPS

print(exp)


	

