# https://www.geeksforgeeks.org/reading-csv-files-in-python/
# https://stackoverflow.com/questions/1624883/alternative-way-to-split-a-list-into-groups-of-n
import pandas as pd
from agents.manager_agent import ManagerAgent
import sys

# Read specific columns from CSV file
columns = ['query']  # Replace with your column names
df = pd.read_csv('/scratch/gpfs/ca2992/TravelPlanner/evaluation/test.csv', usecols=columns)

tools_list = ["notebook", "flights", "attractions", "accommodations",
                  "restaurants", "googleDistanceMatrix", "planner", "cities"]

# Display the DataFrame with specific columns

queries = df['query']
queries = df['query'][0:49]

n = 5
sets = [queries[i:i + n] for i in range(0, len(queries), n)]

print(sets[0])

dir = '/scratch/gpfs/ca2992/EvalData/'

def run_slice_and_write(agent: ManagerAgent, index):
    file_str = dir + "out_" + str(index) +".txt"
    results = []
    for i in range(len(sets[index])):
        request = sets[index][i]
        plan, _ = agent.run(request)
        dict = {}
        dict[request] = plan
        results.append(dict)
        # Clean and reset the agent.
        agent = ManagerAgent(query = "", tools= tools_list)
        if i % 3 == 0: # save some cpu mem
            with open(file_str, mode = "a+") as f:
                for item in results:
                    print("###\n", file = f)
                    print(str(item) + " \n", file = f)
            results = []
    with open(file_str, mode = "a+") as f:
        for item in results:
            print("###\n", file = f)
            print(str(item) + " \n", file = f)
    results = []


def main(args):
    print(args[1], flush=True)
    agent = ManagerAgent(query = "", tools= tools_list)
    run_slice_and_write(agent, int(args[1]))

# TODO: AttributeError: module 'sys' has no attribute 'args'
if __name__ == "__main__":
    main(sys.argv)

file_str = dir + "testing"
dict = {}
dict["request"] = "plan"
results = []
results.append(dict)
with open(file_str, mode = "w") as f:
      print("\n", f)
with open(file_str, mode = "a+") as f:
        for item in results:
            print("###\n", file = f)
            print(str(item) + " \n", file = f)