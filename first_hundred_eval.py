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
init = 0
queries = df['query'][init:99]

n = 10
sets = [queries[i:i + n] for i in range(0, len(queries), n)]

dir = '/scratch/gpfs/ca2992/EvalData/first_hundred/'

def run_slice_and_write(agent: ManagerAgent, index):
    file_str = dir + "out_" + str(index) +".txt"
    results = []
    i = 0
    for val in (sets[index]):
        q_number = init+n*index + i
        request = val
        print("\nrequest in eval ", request, "\n", flush = True)
        plan, _ = agent.run(request)
        dict = {}
        dict[request] = plan
        results.append(dict)
        agent.reset_agent()
        with open(file_str, mode = "a+") as f:
            for item in results:
                print("###\n", file = f)
                print("\nQuery: ", q_number, "\n", file=f)
                print(str(item) + " \n", file = f)
        results = []
        i += 1
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